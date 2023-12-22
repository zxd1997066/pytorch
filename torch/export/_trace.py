import copy
import dataclasses
import functools
import re
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch._dynamo
import torch.fx

import torch.utils._pytree as pytree
from torch._dynamo.exc import UserError, UserErrorType
from torch._export.non_strict_utils import make_constraints, make_fake_inputs
from torch._export.passes.add_runtime_assertions_for_constraints_pass import (
    _AddRuntimeAssertionsForInlineConstraintsPass,
)
from torch._export.passes.collect_tracepoints_pass import CollectTracepointsPass
from torch._export.passes.lift_constant_tensor_pass import lift_constant_tensor_pass
from torch._export.wrappers import _wrap_submodules
from torch._functorch.aot_autograd import aot_export_module, GraphSignature
from torch._guards import detect_fake_mode
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.fx.experimental.symbolic_shapes import (
    ConstraintViolationError,
    GuardOnDataDependentSymNode,
    ShapeEnv,
)
from torch.fx.graph import _PyTreeCodeGen, _PyTreeInfo
from torch.utils._sympy.value_ranges import ValueRangeError

from .dynamic_shapes import _process_constraints, Constraint
from .exported_program import (
    _disable_prexisiting_fake_mode,
    ExportedProgram,
    InputKind,
    ModuleCallEntry,
    ModuleCallSignature,
)
from .graph_signature import (
    _sig_to_specs,
    ArgumentSpec,
    ConstantArgument,
    ExportGraphSignature,
    SymIntArgument,
    TensorArgument,
)


@dataclasses.dataclass
class ExportDynamoConfig:
    """
    Manage Export-specific configurations of Dynamo.
    """

    allow_rnn: bool = True


DEFAULT_EXPORT_DYNAMO_CONFIG = ExportDynamoConfig()


def _convert_input_to_fake(gm, args, kwargs):
    params_buffers = _get_params_buffers(gm)
    fake_inps: List[torch.Tensor] = []
    for node in gm.graph.nodes:
        if node.op == "placeholder" and "val" in node.meta:
            fake_val = node.meta["val"]
            if fake_val is not None and isinstance(fake_val, torch.Tensor):
                fake_inps.append(fake_val)

    if detected_fake_mode := detect_fake_mode(fake_inps):
        fake_mode = detected_fake_mode
    else:
        fake_mode = FakeTensorMode(shape_env=ShapeEnv())

    if len(args) == 0 and len(kwargs) == 0:
        return (), {}, params_buffers, fake_mode

    count = 0

    def convert_to_fake(x):
        nonlocal count
        val = fake_inps[count]
        count += 1
        return val

    fake_args = pytree.tree_map_only(torch.Tensor, convert_to_fake, args)
    # TODO properly use the cached fake tensor
    fake_kwargs = pytree.tree_map_only(torch.Tensor, fake_mode.from_tensor, kwargs)
    fake_params_buffers = pytree.tree_map_only(
        torch.Tensor,
        functools.partial(fake_mode.from_tensor, static_shapes=True),
        params_buffers,
    )
    return fake_args, fake_kwargs, fake_params_buffers, fake_mode


def _replace_param_buffer_names(param_buffer_table, sig):
    for spec in sig.input_specs:
        spec.target = param_buffer_table.get(spec.target, spec.target)
    for spec in sig.output_specs:
        spec.target = param_buffer_table.get(spec.target, spec.target)


def _reorder_kwargs_by_names(
    arg_names: List[str], args: Tuple[Any], kwargs: Dict[str, Any]
):
    assert len(arg_names) == len(args) + len(kwargs), (
        f"Total number of arg names is expected to be {len(arg_names)} "
        f"but got {len(args)} positional args, {len(kwargs)} kwargs."
    )
    return OrderedDict({kw_name: kwargs[kw_name] for kw_name in arg_names[len(args) :]})


def _normalize_nn_module_stack(gm_torch_level, root_cls):
    # Append a root module to every nn_module_stack.
    root = "L['self']"
    root_key = re.sub(r"[^a-zA-Z0-9]", "_", root)
    for gm in gm_torch_level.modules():
        if not isinstance(gm, torch.fx.GraphModule):
            continue
        for node in gm.graph.nodes:
            if node.op in ["placeholder", "output"]:
                continue
            add_root = True
            if nn_module_stack := node.meta.get("nn_module_stack", {}):
                path, ty = next(iter(nn_module_stack.values()))
                assert issubclass(ty, torch.nn.Module)
                # TODO Figure out why sometimes we have root sometimes we don't.
                if path == root and ty is root_cls:
                    add_root = False
            if add_root:

                def normalize_path(path):
                    try:
                        parts = []

                        class Path:
                            def __getattr__(self, name):
                                parts.append(name)
                                return self

                            def __getitem__(self, idx):
                                parts.append(str(idx))
                                return self

                        eval(path, {"L": {"self": Path()}})
                        return ".".join(parts)
                    except Exception:  # TODO(zhxchen17) Remove this.
                        return path

                nn_module_stack = {root_key: (root, root_cls), **nn_module_stack}
                node.meta["nn_module_stack"] = {
                    key: (normalize_path(path), ty)
                    for key, (path, ty) in nn_module_stack.items()
                }


def _get_param_buffer_mapping(
    original_module: torch.nn.Module,
    traced_module: torch.nn.Module,
) -> Dict[str, str]:
    """
    Returns a mapping of parameter/buffer names from the new module to the
    original model. This is to help with restoring the FQN for parameter/buffers
    of a traced module to what the original module contains.
    """

    param_lookup: Dict[int, List[str]] = {}
    buffer_lookup: Dict[int, List[str]] = {}
    for name, param in original_module.named_parameters(remove_duplicate=False):
        param_lookup.setdefault(id(param), []).append(name)
    for name, buffer in original_module.named_buffers(remove_duplicate=False):
        buffer_lookup.setdefault(id(buffer), []).append(name)

    param_buffer_table: Dict[str, str] = {}
    for dynamo_name, dynamo_param in traced_module.named_parameters(
        remove_duplicate=False
    ):
        assert dynamo_name not in param_buffer_table
        if id(dynamo_param) in param_lookup:
            param_buffer_table[dynamo_name] = param_lookup[id(dynamo_param)].pop()

    for dynamo_name, dynamo_buffer in traced_module.named_buffers(
        remove_duplicate=False
    ):
        assert dynamo_name not in param_buffer_table
        if id(dynamo_buffer) in buffer_lookup:
            param_buffer_table[dynamo_name] = buffer_lookup[id(dynamo_buffer)].pop()

    return param_buffer_table


def _restore_state_dict(
    original_module: torch.nn.Module, traced_module: torch.fx.GraphModule
) -> None:
    """
    Restores the state dict of the traced module to that of the original module.
    """
    param_buffer_table = _get_param_buffer_mapping(original_module, traced_module)
    # Since the graph module is flattened (no module heirarchy), we
    # need to noramlize the module by replacing "." with "_". If we
    # don't, it will try to save the weight to a submodule which no
    # longer exists.
    for name, fqn in param_buffer_table.items():
        param_buffer_table[name] = fqn.replace(".", "_")

    # Replace state dict attr names with the fqn
    for name, fqn in param_buffer_table.items():
        if not hasattr(traced_module, name):
            continue

        attr = getattr(traced_module, name)
        if isinstance(attr, torch.Tensor) and not isinstance(attr, torch.nn.Parameter):
            traced_module.register_buffer(fqn, attr)
        else:
            setattr(traced_module, fqn, attr)
        delattr(traced_module, name)

    # Replace graph getattr nodes with the correct name
    for node in traced_module.graph.nodes:
        if node.op == "get_attr":
            attr_name = node.target
            if attr_name in param_buffer_table:
                node.target = param_buffer_table[attr_name]

    traced_module.recompile()


def _export_to_torch_ir(
    f: Callable,
    args: Tuple[Any, ...],
    kwargs: Optional[Dict[str, Any]] = None,
    constraints: Optional[List[Constraint]] = None,
    *,
    preserve_module_call_signature: Tuple[str, ...] = (),
    disable_constraint_solver: bool = False,
    restore_fqn: bool = True,
) -> torch.fx.GraphModule:
    """
    Traces either an nn.Module's forward function or just a callable with PyTorch
    operations inside and produce a torch.fx.GraphModule in torch IR.
    """

    constraints = constraints or []
    kwargs = kwargs or {}

    if not isinstance(args, tuple):
        raise UserError(
            UserErrorType.INVALID_INPUT,
            f"Expecting `args` to be a tuple of example positional inputs, got {type(args)}",
        )

    # We convert to nn.Module because __call__ of ExportedProgram
    # is untracable right now.
    if isinstance(f, ExportedProgram):
        f = f.module()

    with torch._dynamo.config.patch(dataclasses.asdict(DEFAULT_EXPORT_DYNAMO_CONFIG)):
        try:
            module_call_specs: Dict[str, Dict[str, pytree.TreeSpec]] = {}
            with _wrap_submodules(f, preserve_module_call_signature, module_call_specs):
                gm_torch_level, _ = torch._dynamo.export(
                    f,
                    constraints=constraints,
                    assume_static_by_default=True,
                    tracing_mode="symbolic",
                    disable_constraint_solver=disable_constraint_solver,
                )(
                    *args,
                    **kwargs,
                )
        except (ConstraintViolationError, ValueRangeError) as e:
            raise UserError(UserErrorType.CONSTRAINT_VIOLATION, str(e))  # noqa: TRY200
        except GuardOnDataDependentSymNode as e:
            raise UserError(  # noqa: TRY200
                UserErrorType.ANTI_PATTERN,
                f"Consider annotating your code using torch._constrain_as_*(). {str(e)}",
                case_name="constrain_as_size_example",
            )

    gm_torch_level.meta["module_call_specs"] = module_call_specs

    if isinstance(f, torch.nn.Module) and restore_fqn:
        _restore_state_dict(f, gm_torch_level)

    return gm_torch_level


def _unlift_user_inputs_to_buffers(
    gm_torch_level: torch.fx.GraphModule, aot_export_args
) -> List[str]:
    flat_args = pytree.tree_leaves(aot_export_args)
    user_input_names = []
    with gm_torch_level.graph.inserting_before():
        for i, (arg, node) in enumerate(zip(flat_args, gm_torch_level.graph.nodes)):
            assert node.op == "placeholder"
            user_input_names.append(node.name)
            if isinstance(arg, torch.Tensor):
                assert not hasattr(gm_torch_level, node.name)
                gm_torch_level.register_buffer(node.name, arg)
                get_attr = gm_torch_level.graph.get_attr(node.name)
                node.replace_all_uses_with(get_attr)
                get_attr.meta = copy.copy(node.meta)

    for node in list(gm_torch_level.graph.nodes):
        if node.op == "placeholder":
            assert len(node.users) == 0
            gm_torch_level.graph.erase_node(node)
    gm_torch_level.recompile()
    return user_input_names


def _lift_buffers_to_user_inputs(
    gm: torch.fx.GraphModule,
    graph_signature: GraphSignature,
    user_input_names: List[str],
) -> Dict[str, str]:
    assert len(graph_signature.user_inputs) == 0
    assert graph_signature.backward_signature is None
    names = set(user_input_names)

    placeholders = [node for node in gm.graph.nodes if node.op == "placeholder"]
    # user inputs are always added in the end
    start = len(graph_signature.parameters)
    end = start + len(graph_signature.buffers)
    buffer_nodes = placeholders[start:end]
    last_placeholder_node = placeholders[-1] if len(placeholders) > 0 else None
    old_nodes: Dict[str, torch.fx.Node] = {}
    for node in buffer_nodes:
        buffer_name = graph_signature.inputs_to_buffers[node.name]
        if buffer_name not in names:
            continue
        old_nodes[buffer_name] = node
    replaces = {}
    new_node_names: Dict[str, str] = {}
    with gm.graph.inserting_after(last_placeholder_node):
        for name in reversed(user_input_names):
            new_node = gm.graph.placeholder(name)
            new_node.target = new_node.name
            new_node_names[name] = new_node.name
            if name in old_nodes:
                old_node = old_nodes[name]
                new_node.meta = copy.copy(old_node.meta)
                old_node.replace_all_uses_with(new_node)
                replaces[old_node.name] = new_node.name
    new_node_names = dict(reversed(new_node_names.items()))
    for old_node in old_nodes.values():
        gm.graph.erase_node(old_node)

    gm.recompile()

    graph_signature.buffers = [b for b in graph_signature.buffers if b not in names]
    graph_signature.inputs_to_buffers = {
        i: b for i, b in graph_signature.inputs_to_buffers.items() if b not in names
    }
    user_inputs_to_mutate = {
        o: b for o, b in graph_signature.buffers_to_mutate.items() if b in names
    }
    graph_signature.buffers_to_mutate = {
        o: b for o, b in graph_signature.buffers_to_mutate.items() if b not in names
    }
    graph_signature.user_inputs.extend(new_node_names.values())  # type: ignore[arg-type]
    graph_signature.user_outputs = [
        replaces[o] if o in replaces else o for o in graph_signature.user_outputs
    ]
    return user_inputs_to_mutate  # type: ignore[return-value]


def _export_non_strict(
    mod,
    fake_args,
    fake_kwargs,
    fake_params_buffers,
    *,
    transform=lambda x: x,  # TODO(zhxchen17) Revisit if this is needed later.
):
    # This _reparametrize_module makes sure inputs and module.params/buffers have the same fake_mode,
    # otherwise aot_export_module will error out because it sees a mix of fake_modes.
    # And we want aot_export_module to use the fake_tensor mode in dynamo to keep the pipeline easy to reason about.
    with torch.nn.utils.stateless._reparametrize_module(mod, fake_params_buffers):
        gm, graph_signature = transform(aot_export_module)(
            mod, (*fake_args, *fake_kwargs.values()), trace_joint=False
        )

    # NOTE: aot_export adds symint metadata for placeholders with int values;
    # since these become specialized, we replace such metadata with the original values
    flat_args = pytree.tree_leaves((fake_args, fake_kwargs))
    index = 0
    total_param_buffers = len(graph_signature.parameters) + len(graph_signature.buffers)
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            if index >= total_param_buffers:
                user_arg = flat_args[index - total_param_buffers]
                if not isinstance(user_arg, torch.Tensor):
                    node.meta["val"] = user_arg
            index += 1

    is_joint = graph_signature.backward_signature is not None

    def make_argument_spec(node) -> ArgumentSpec:
        assert "val" in node.meta, f"{node} has no 'val' metadata field"
        val = node.meta["val"]
        if isinstance(val, FakeTensor):
            return TensorArgument(name=node.name)
        elif isinstance(val, torch.SymInt):
            return SymIntArgument(name=node.name)
        else:
            return ConstantArgument(value=val)

    input_specs, output_specs = _sig_to_specs(
        user_inputs=set(graph_signature.user_inputs),
        inputs_to_parameters=graph_signature.inputs_to_parameters,  # type: ignore[arg-type]
        inputs_to_buffers=graph_signature.inputs_to_buffers,  # type: ignore[arg-type]
        user_outputs=set(graph_signature.user_outputs),  # type: ignore[arg-type]
        buffer_mutations=graph_signature.buffers_to_mutate,  # type: ignore[arg-type]
        user_input_mutations=gm.meta.get("user_inputs_to_mutate", {}),  # type: ignore[arg-type]
        grad_params=graph_signature.backward_signature.gradients_to_parameters if is_joint else {},  # type: ignore[arg-type, union-attr]
        grad_user_inputs=graph_signature.backward_signature.gradients_to_user_inputs if is_joint else {},  # type: ignore[arg-type, union-attr]
        loss_output=graph_signature.backward_signature.loss_output if is_joint else None,  # type: ignore[arg-type, union-attr]
        inputs=[
            make_argument_spec(node)
            for node in gm.graph.nodes
            if node.op == "placeholder"
        ],
        outputs=[
            make_argument_spec(node)
            for node in pytree.tree_leaves(next(iter(reversed(gm.graph.nodes))).args)
        ],
    )
    export_graph_signature = ExportGraphSignature(
        input_specs=input_specs, output_specs=output_specs
    )

    tensor_constants = lift_constant_tensor_pass(gm, export_graph_signature)

    @dataclasses.dataclass
    class _ExportedProgramNonStrict:
        gm: torch.fx.GraphModule
        sig: ExportGraphSignature
        tensor_constants: Dict[str, torch.Tensor]

    return _ExportedProgramNonStrict(
        gm,
        export_graph_signature,
        tensor_constants,
    )


def _get_params_buffers(mod: torch.nn.Module) -> Dict[str, torch.Tensor]:
    params_buffers: Dict[str, torch.Tensor] = {}
    for name, param in mod.named_parameters(remove_duplicate=False):
        params_buffers[name] = param

    for name, buffer in mod.named_buffers(remove_duplicate=False):
        params_buffers[name] = buffer
    return params_buffers


@_disable_prexisiting_fake_mode
def _export(
    f: Callable,
    args: Tuple[Any, ...],
    kwargs: Optional[Dict[str, Any]] = None,
    constraints: Optional[List[Constraint]] = None,
    *,
    strict: bool = True,
    preserve_module_call_signature: Tuple[str, ...] = (),
) -> ExportedProgram:
    """
    Traces either an nn.Module's forward function or just a callable with PyTorch
    operations inside and produce a ExportedProgram.

    Args:
        m: the `nn.Module` or callable to trace.

        args: example positional inputs.

        kwargs: optional example keyword inputs.

        constraints: A optional list of constraints on the dynamic arguments specifying
            their possible range of their shapes

        preserve_module_call_signature: A list of submodule paths for which the original
            calling conventions are preserved as metadata.

    Returns:
        An ExportedProgram containing the traced method.
    """
    constraints = constraints or []
    kwargs = kwargs or {}

    if not strict:
        assert isinstance(f, torch.nn.Module)
        assert len(preserve_module_call_signature) == 0
        assert len(kwargs) == 0, "keyword arguments NYI"
        out_spec = None

        def _tuplify_outputs(aot_export):
            def _aot_export_non_strict(mod, args, **kwargs):
                class Wrapper(torch.nn.Module):
                    def __init__(self, mod):
                        super().__init__()
                        self._export_root = mod

                    def forward(self, *args, **kwargs):
                        nonlocal out_spec
                        flat_outs, out_spec = pytree.tree_flatten(
                            self._export_root(*args, **kwargs)
                        )
                        return tuple(flat_outs)

                gm, sig = aot_export(Wrapper(mod), args, **kwargs)

                def strip_root(x):
                    if isinstance(x, str) and x.startswith("_export_root"):
                        stripped = x[len("_export_root") :]
                        return stripped[1:] if stripped.startswith(".") else stripped
                    return x

                def fixup_key(x):
                    return "L__self__" + strip_root(x)

                sig.parameters = pytree.tree_map(strip_root, sig.parameters)
                sig.buffers = pytree.tree_map(strip_root, sig.buffers)
                sig.inputs_to_buffers = pytree.tree_map(
                    strip_root, sig.inputs_to_buffers
                )
                sig.inputs_to_parameters = pytree.tree_map(
                    strip_root, sig.inputs_to_parameters
                )
                sig.buffers_to_mutate = pytree.tree_map(
                    strip_root, sig.buffers_to_mutate
                )
                for node in gm.graph.nodes:
                    if "nn_module_stack" in node.meta:
                        nn_module_stack = node.meta["nn_module_stack"]
                        # Delete the wrapper module reference
                        del nn_module_stack[""]
                        node.meta["nn_module_stack"] = {
                            fixup_key(key): val
                            for key, val in pytree.tree_map(
                                strip_root, nn_module_stack
                            ).items()
                        }

                return gm, sig

            return _aot_export_non_strict

        fake_mode, fake_args, src_equalities, original_signature = make_fake_inputs(
            f, args, constraints
        )
        ep_non_strict = _export_non_strict(
            f, fake_args, {}, f.state_dict(), transform=_tuplify_outputs
        )
        range_constraints, equality_constraints = make_constraints(
            fake_mode, src_equalities, original_signature, ep_non_strict.gm
        )
        assert out_spec is not None
        return ExportedProgram(
            ep_non_strict.gm,
            ep_non_strict.gm.graph,
            ep_non_strict.sig,
            _get_params_buffers(f),
            range_constraints,
            equality_constraints,
            [
                ModuleCallEntry(
                    "",
                    ModuleCallSignature(
                        [], [], pytree.tree_flatten((args, {}))[1], out_spec
                    ),
                )
            ],
            (args, kwargs),
            tensor_constants=ep_non_strict.tensor_constants,
        )

    gm_torch_level = _export_to_torch_ir(
        f,
        args,
        kwargs,
        constraints,
        preserve_module_call_signature=preserve_module_call_signature,
        restore_fqn=False,  # don't need to restore because we will do it later
    )

    params_buffers = _get_params_buffers(gm_torch_level)

    # We detect the fake_mode by looking at gm_torch_level's placeholders, this is the fake_mode created in dynamo.
    (
        fake_args,
        fake_kwargs,
        fake_params_buffers,
        dynamo_fake_mode,
    ) = _convert_input_to_fake(gm_torch_level, args, kwargs)

    # First, we want to pass through the graph to try populating
    # val field for getattr if there is anything missing.
    # THis can happen when quantization adds extra params and forgets
    # to update "val"
    for node in gm_torch_level.graph.nodes:
        if node.op == "get_attr" and "val" not in node.meta:
            attr = getattr(gm_torch_level, node.target)
            # Checks if it is not a HigherOrderOp branch or a module
            if not isinstance(attr, torch.nn.Module):
                assert (
                    dynamo_fake_mode is not None
                ), "Cannot find dynamo_fake_mode. This could be due to the exported graph module have no placeholders."
                node.meta["val"] = dynamo_fake_mode.from_tensor(
                    attr, static_shapes=True
                )

    # When aot_export lifts the params, we lose the nn_module_stack
    # and source_fn from the param nodes as they are treated as fresh inputs
    # Therefore, we manually extract them before calling into aot_export
    params_buffers_to_node_meta = {}
    for node in gm_torch_level.graph.nodes:
        target = node.target
        meta = node.meta
        if node.op == "call_module":
            submodule = getattr(gm_torch_level, target)
            if isinstance(submodule, torch.nn.Module):
                for name, _ in submodule.named_parameters(
                    recurse=True, remove_duplicate=False
                ):
                    params_buffers_to_node_meta[target + "." + name] = meta

                for name, _ in submodule.named_buffers(
                    recurse=True, remove_duplicate=False
                ):
                    params_buffers_to_node_meta[target + "." + name] = meta

        if node.op == "get_attr":
            submodule = getattr(gm_torch_level, target)
            if not isinstance(submodule, torch.fx.GraphModule):
                params_buffers_to_node_meta[target] = meta

        # If the call_function uses param as input, we also need to update params' meta
        # with this call_function node's meta.
        # This is basically the same flow as torch.fx.traceback.preserve_meta()
        if node.op == "call_function" and not isinstance(
            node.target, torch._ops.HigherOrderOperator
        ):
            for arg in node._input_nodes:
                if arg.op == "get_attr":
                    for entry in torch.fx.proxy._COPY_META_FIELDS:
                        if entry in meta:
                            params_buffers_to_node_meta[arg.target][entry] = meta[entry]

    # Fix the graph output signature to be tuple if scalar
    out_spec = orig_out_spec = gm_torch_level._out_spec
    assert out_spec is not None
    # aot_export expect the return type to always be a tuple.
    if out_spec.type not in (list, tuple):
        out_spec = pytree.TreeSpec(tuple, None, [out_spec])

    orig_args = gm_torch_level.graph._codegen.pytree_info.orig_args  # type: ignore[attr-defined]

    gm_torch_level.graph._codegen = _PyTreeCodeGen(
        _PyTreeInfo(
            orig_args,
            gm_torch_level._in_spec,
            out_spec,
        )
    )
    gm_torch_level.recompile()

    # Restore FQN of param/buffers
    param_buffer_table: Dict[str, str] = (
        _get_param_buffer_mapping(f, gm_torch_level)
        if isinstance(f, torch.nn.Module)
        else {}
    )

    if isinstance(f, torch.nn.Module):
        _normalize_nn_module_stack(gm_torch_level, type(f))

    def _process_user_inputs(aot_export):
        def _aot_export_strict(gm_torch_level: torch.fx.GraphModule, args, **kwargs):
            user_input_names = _unlift_user_inputs_to_buffers(gm_torch_level, args)
            gm, graph_signature = aot_export(gm_torch_level, (), **kwargs)
            user_inputs_to_mutate = _lift_buffers_to_user_inputs(
                gm, graph_signature, user_input_names
            )
            # TODO unfortunately preserving graph-level metadata is not
            # working well with aot_export. So we manually copy it.
            # (The node-level meta is addressed above.)
            gm.meta.update(gm_torch_level.meta)
            assert "user_inputs_to_mutate" not in gm.meta
            gm.meta["user_inputs_to_mutate"] = user_inputs_to_mutate
            return gm, graph_signature

        return _aot_export_strict

    # Note: aot_export_module doesn't accept kwargs, we'd like to reorder the kwargs as an OrderedDict
    # to follow the order in orig_args and correctly call module
    ep_non_strict = _export_non_strict(
        gm_torch_level,
        fake_args,
        _reorder_kwargs_by_names(orig_args, fake_args, fake_kwargs),
        fake_params_buffers,
        transform=_process_user_inputs,
    )

    gm = ep_non_strict.gm
    export_graph_signature = ep_non_strict.sig
    tensor_constants = ep_non_strict.tensor_constants

    # After aot_export, set the param/buffer metadata back into placeholders
    # Technically, users can still construct this data from param names
    # without relying on this metadata
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            if node.target in export_graph_signature.inputs_to_parameters:
                param_name = export_graph_signature.inputs_to_parameters[node.target]
                if param_name in params_buffers_to_node_meta:
                    for k, v in params_buffers_to_node_meta[param_name].items():
                        node.meta[k] = v
            if node.target in export_graph_signature.inputs_to_buffers:
                buffer_name = export_graph_signature.inputs_to_buffers[node.target]
                if buffer_name in params_buffers_to_node_meta:
                    for k, v in params_buffers_to_node_meta[buffer_name].items():
                        node.meta[k] = v

    # The unbacked symint symbols are updated in aot_export
    # so we serialize them here instead of inside dynamo

    gm.meta["inline_constraints"] = {
        k: v
        for k, v in dynamo_fake_mode.shape_env.runtime_var_to_range.items()
        if re.match(r"^[if]\d+$", str(k))
    }

    num_lifted = next(
        (
            i
            for i, s in enumerate(export_graph_signature.input_specs)
            if s.kind == InputKind.USER_INPUT
        ),
        len(export_graph_signature.input_specs),
    )
    flat_args, orig_in_spec = pytree.tree_flatten((args, kwargs))
    range_constraints, equality_constraints = _process_constraints(
        gm,
        num_lifted,
        flat_args,
    )

    if isinstance(f, torch.nn.Module):
        _replace_param_buffer_names(param_buffer_table, export_graph_signature)
        params_buffers = {
            param_buffer_table.get(name, name): tensor
            for name, tensor in params_buffers.items()
        }

    module_call_signatures = {
        fqn: ModuleCallSignature(inputs=[], outputs=[], **specs)
        for fqn, specs in gm_torch_level.meta["module_call_specs"].items()
    }

    if len(preserve_module_call_signature) > 0:
        res = CollectTracepointsPass(module_call_signatures, export_graph_signature)(gm)
        assert res is not None
        gm = res.graph_module

    assert orig_out_spec is not None
    exported_program = ExportedProgram(
        gm,
        gm.graph,
        export_graph_signature,
        # TODO(zhxchen17) Return empty state_dict for functions.
        params_buffers,
        range_constraints,
        equality_constraints,
        [
            ModuleCallEntry(
                "",
                ModuleCallSignature(
                    inputs=[], outputs=[], in_spec=orig_in_spec, out_spec=orig_out_spec
                ),
            )
        ]
        + [ModuleCallEntry(fqn, sig) for fqn, sig in module_call_signatures.items()],
        (args, kwargs),
        tensor_constants=tensor_constants,
    )

    if len(range_constraints) > 0 or len(equality_constraints) > 0:
        exported_program = exported_program._transform(
            _AddRuntimeAssertionsForInlineConstraintsPass(
                range_constraints, equality_constraints
            )
        )

    return exported_program
