import copy
import gc
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch

# Deterministic behavior was enabled with either torch.use_deterministic_algorithms(True)
# or at::Context::setDeterministicAlgorithms(true), but this operation is not deterministic
# because it uses CuBLAS and you have CUDA >= 10.2
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

TensorCollection = Union[Dict[str, torch.Tensor], List[torch.Tensor], torch.Tensor]
# pyre-ignore [33]
NumericCheckInput = Tuple[Any, torch.nn.Module, Optional[Dict[str, str]]]
NumericCheckOutput = Dict[str, Any]


def check_structured_tensor_isclose(
    original: TensorCollection,
    new: TensorCollection,
    rtol: float = 1e-04,
    atol: float = 1e-04,
    allow_new: bool = False,
) -> Tuple[bool, str]:
    """
    Helper method to check if generated model output matches with the original model.

    Args:
        original(TensorCollection): original model output
        transformed(TensorCollection): generated model output
        rtol(float = 1e-04): relative tolerance
        atol(float = 1e-04): absolute tolerance

    Returns:
        bool
    """
    if isinstance(original, Mapping):
        if not isinstance(new, Mapping):
            return (False, f"original is a dict, but new is not: {type(new)}")
        if set(original.keys()) != set(new.keys()):
            not_in_new = set(original.keys()) - (set(new.keys()))
            not_in_old = set(new.keys()) - (set(original.keys()))
            if allow_new:
                if not_in_new:
                    return (
                        False,
                        f"keys mismatches: {not_in_new} in original but not in new.",
                    )
            else:
                return (
                    False,
                    f"keys mismatches: {not_in_new} in original but not in new, {not_in_old} in new but not in original.",
                )
        for k in original:
            if not (
                torch.isclose(original[k], new[k], rtol=rtol, atol=atol, equal_nan=True)
                .all()
                .item()
            ):
                return (
                    False,
                    f"tensor from key {k} mismatches: original: {original[k]}, new: {new[k]}",
                )
        return (True, "")
    elif isinstance(original, Sequence):
        if not isinstance(new, Sequence):
            return (False, f"original is a list, but new is not: {type(new)}")
        if len(original) != len(new):
            return (
                False,
                f"length mismatches: original: {len(original)}, new: {len(new)}",
            )
        for i, (o, t) in enumerate(zip(original, new)):
            if not (
                torch.isclose(o, t, rtol=rtol, atol=atol, equal_nan=True).all().item()
            ):
                return (
                    False,
                    f"tensor from indices {i} mismatches: original: {original[i]}, new: {new[i]}",
                )
        return (True, "")
    elif isinstance(original, torch.Tensor):
        if not isinstance(new, torch.Tensor):
            return (False, f"original is a tensor, but new is not: {type(new)}")
        if not (
            torch.isclose(original, new, rtol=rtol, atol=atol, equal_nan=True)
            .all()
            .item()
        ):
            return (
                False,
                f"tensor mismatches: original: {original}, new: {new}",
            )
        return (True, "")

    else:
        raise ValueError("Unsupported input to check_structured_tensor_isclose")


class NumericChecker(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def stage(self) -> str:
        pass

    @abstractmethod
    # pyre-ignore [2]
    def check(self, a, b, **kwargs) -> Tuple[bool, str]:
        pass

    @abstractmethod
    # pyre-ignore [3]
    # pyre-ignore [2]
    def extract(self, model, output, param_remap):
        pass

    def __repr__(self) -> str:
        return f"({repr(self.name)})"


class AllCloseTensorChecker(NumericChecker):

    # pyre-ignore [2]
    def check(self, a: Any, b: Any, **kwargs) -> Tuple[bool, str]:
        rtol = kwargs["rtol"] if "rtol" in kwargs else 1e-4
        atol = kwargs["atol"] if "atol" in kwargs else 1e-4
        assert type(rtol) == float and type(atol) == float
        return check_structured_tensor_isclose(a, b, rtol=rtol, atol=atol)


class EqualChecker(NumericChecker):
    # pyre-ignore [2]
    def check(self, a, b, **kwargs) -> Tuple[bool, str]:
        not_in_new = set(a) - set(b)
        not_in_old = set(b) - set(a)
        return (
            set(a) == set(b),
            f"mismatches: {not_in_new} in original but not in new, {not_in_old} in new but not in original."
            if a != b
            else "",
        )


class OutputChecker(AllCloseTensorChecker):
    @property
    def name(self) -> str:
        return "output"

    @property
    def stage(self) -> str:
        return "fwd"

    # pyre-ignore [2]
    # pyre-ignore [3]
    def extract(self, model, output, param_remap) -> Any:
        return output


class ParameterChecker(AllCloseTensorChecker):
    @property
    def name(self) -> str:
        return "parameter"

    @property
    def stage(self) -> str:
        return "optimizer"

    # pyre-ignore [2]
    # pyre-ignore [3]
    def extract(self, model, output, param_remap) -> Any:
        parameters = {}
        names = [name for name, _ in model.named_parameters()]
        remapped_keys, _ = remap_keys(model, names, param_remap)
        for param_name, value in model.named_parameters():
            if param_name in remapped_keys:
                parameters[remapped_keys[param_name]] = value
        return parameters


class BufferChecker(AllCloseTensorChecker):
    def __init__(self) -> None:
        super().__init__()

    @property
    def name(self) -> str:
        return "buffer"

    @property
    def stage(self) -> str:
        return "fwd"

    # pyre-ignore [2]
    # pyre-ignore [3]
    def extract(self, model, output, param_remap) -> Any:
        try:
            buffers = model.named_buffers()
            mapped_buffers = {}
            names = [name for name, buffer in buffers.items()]
            remapped_keys, _ = remap_keys(model, names, param_remap)
            for param_name, value in buffers.items():
                if param_name in remapped_keys:
                    mapped_buffers[remapped_keys[param_name]] = value
            return mapped_buffers
        except Exception as e:
            logging.info(f"Failed to get buffer: {e}")
            return {}

    # pyre-ignore [2]
    def check(self, a: Any, b: Any, **kwargs) -> Tuple[bool, str]:
        rtol = kwargs["rtol"] if "rtol" in kwargs else 1e-4
        atol = kwargs["atol"] if "atol" in kwargs else 1e-4
        assert type(rtol) == float and type(atol) == float
        # we allow new tensors showing in the new model
        return check_structured_tensor_isclose(
            a, b, rtol=rtol, atol=atol, allow_new=True
        )


class GradientChecker(AllCloseTensorChecker):
    def __init__(self) -> None:
        super().__init__()

    @property
    def name(self) -> str:
        return "gradient"

    @property
    def stage(self) -> str:
        return "bwd"

    # pyre-ignore [2]
    # pyre-ignore [3]
    def extract(self, model, output, param_remap) -> Any:
        parameters = model.named_parameters()
        names = [name for name, p in parameters.items() if p.grad is not None]
        remapped_keys, _ = remap_keys(model, names, param_remap)
        gradients = {}
        for key, p in parameters.items():
            grad = p.grad
            if grad is not None and key in remapped_keys:
                gradients[remapped_keys[key]] = grad.clone().detach()
        return gradients


class NumericStatus(Enum):
    FAILED = 0
    PASSED = 1


@dataclass
class CheckSettings:
    """Class for numeric check settings."""

    rtol: float = 1e-4
    atol: float = 1e-4
    iteration: int = 1
    checklist: List[NumericChecker] = field(default_factory=list)
    passlist: List[NumericChecker] = field(default_factory=list)


class CheckResult:
    """Class for numeric check result."""

    def __init__(self, status: NumericStatus, settings: CheckSettings) -> None:
        self.status = status
        self.breakdowns: Dict[str, List[bool]] = {}
        self.explainations: List[str] = []
        self.settings = settings

    def add_breakdown(self, name: str, status: bool) -> None:
        if name not in self.breakdowns:
            self.breakdowns[name] = []
        self.breakdowns[name].append(status)

    def __repr__(self) -> str:
        return f"({repr(self.status), repr(self.breakdowns)})"

    def get_first_failed_checker(self) -> Optional[str]:
        for checker_to_pass in self.settings.passlist:
            name = checker_to_pass.name
            if name in self.breakdowns and False in self.breakdowns[name]:
                return name
        return None


OUTPUT_CHECKER = OutputChecker()
PARAMETER_CHECKER = ParameterChecker()
BUFFER_CHECKER = BufferChecker()
GRADIENT_CHECKER = GradientChecker()

FWD_NORMAL_CHECK = CheckSettings(rtol=1e-4, atol=1e-4, checklist=[OUTPUT_CHECKER])
FWD_STRICT_CHECK = CheckSettings(
    rtol=1e-4,
    atol=1e-4,
    checklist=[OUTPUT_CHECKER, BUFFER_CHECKER],
    passlist=[OUTPUT_CHECKER, BUFFER_CHECKER],
)
FWD_BWD_WEAK_CHECK = CheckSettings(
    rtol=1e-4,
    atol=1e-4,
    checklist=[OUTPUT_CHECKER, BUFFER_CHECKER, GRADIENT_CHECKER],
    passlist=[OUTPUT_CHECKER],
)
FWD_BWD_NORMAL_CHECK = CheckSettings(
    rtol=1e-4,
    atol=1e-4,
    checklist=[OUTPUT_CHECKER, BUFFER_CHECKER, GRADIENT_CHECKER],
    passlist=[OUTPUT_CHECKER, GRADIENT_CHECKER],
)
FWD_BWD_STRICT_CHECK = CheckSettings(
    rtol=1e-4,
    atol=1e-4,
    checklist=[OUTPUT_CHECKER, BUFFER_CHECKER, GRADIENT_CHECKER],
    passlist=[OUTPUT_CHECKER, BUFFER_CHECKER, GRADIENT_CHECKER],
)
E2E_STRICT_CHECK = CheckSettings(
    rtol=1e-4,
    atol=1e-4,
    checklist=[OUTPUT_CHECKER, PARAMETER_CHECKER, BUFFER_CHECKER],
    passlist=[OUTPUT_CHECKER, PARAMETER_CHECKER, BUFFER_CHECKER],
    iteration=1,
)

MAIN_RANDOM_SEED = 1337


def set_deterministic() -> None:
    """Make torch manual seed deterministic."""
    import random

    import numpy
    import torch

    torch.manual_seed(MAIN_RANDOM_SEED)
    random.seed(MAIN_RANDOM_SEED)
    numpy.random.seed(MAIN_RANDOM_SEED)
    torch.use_deterministic_algorithms(True)


def clean_memory() -> None:
    """Clean memory to avoid OOM."""
    gc.collect()
    torch.cuda.empty_cache()


def compare_result(
    expected: Dict[str, Any],
    actual: Dict[str, Any],
    settings: CheckSettings,
) -> CheckResult:
    check_result = CheckResult(NumericStatus.PASSED, settings=settings)
    for i in range(settings.iteration):
        for checker in settings.checklist:
            explain = ""
            if checker.name in expected and checker.name in actual:
                if i < len(expected[checker.name]) and i < len(actual[checker.name]):
                    current_iter_result, explain = checker.check(
                        expected[checker.name][i],
                        actual[checker.name][i],
                        rtol=settings.rtol,
                        atol=settings.atol,
                    )
                    check_result.add_breakdown(checker.name, current_iter_result)
                    if current_iter_result:
                        logging.info(
                            f"Numeric check passed for {checker.name} at iter {i}."
                        )
                        continue
            else:
                explain = f"{checker.name} in expected={checker.name in expected}, {checker.name} in actual={checker.name in actual}"
            if checker in settings.passlist:
                check_result.status = NumericStatus.FAILED
                prefix = "[Required]"
            else:
                prefix = "[Not required]"
            check_result.explainations.append(
                f"Numeric check failed for {checker.name} at iter {i}, reason: {explain}"
            )
            logging.error(
                f"{prefix} Numeric check failed for {checker.name} at iter {i}."
            )
        if check_result.status == NumericStatus.FAILED:
            break
    if "exception" in actual:
        check_result.explainations.append(f"Exception raised: {actual['exception']}")
    logging.info(f"Final numeric check result: {check_result}")
    return check_result


def remap_keys(
    # pyre-ignore [2]
    model: Any,
    original_keys: List[str],
    params_remap: Optional[Dict[str, str]] = None,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    reverted_map = {}
    full_remap = {}
    for key in original_keys:
        if params_remap and key in params_remap:
            new_param_key = params_remap[key]
        else:
            new_param_key = "_orig_mod." + key
        full_remap[key] = new_param_key
        reverted_map[new_param_key] = key
    return (full_remap, reverted_map)


def reset_actual_model_buffer(
    # pyre-ignore [2]
    gt_model: Any,
    # pyre-ignore [2]
    actual_model: Any,
    gt_param_remap: Optional[Dict[str, str]] = None,
    actual_params_remap: Optional[Dict[str, str]] = None,
) -> None:
    gt_buffers = dict(gt_model.named_buffers())
    gt_buffer_names = [name for name, _ in gt_buffers.items()]
    actual_buffer_names = [name for name, _ in actual_model.named_buffers()]
    gt_remapped_keys, _ = remap_keys(gt_model, gt_buffer_names, gt_param_remap)
    _, actual_reverted_remapped_key = remap_keys(
        actual_model, actual_buffer_names, actual_params_remap
    )
    to_be_updated = actual_model.state_dict()
    for buffer_key, buffer_value in gt_buffers.items():
        if buffer_key in gt_remapped_keys:
            remapped_key = gt_remapped_keys[buffer_key]
            if remapped_key in actual_reverted_remapped_key:
                actual_model_buffer_key = actual_reverted_remapped_key[remapped_key]
                to_be_updated[actual_model_buffer_key] = buffer_value.clone().detach()
    actual_model.load_state_dict(to_be_updated)


def numeric_check(
    actual_model_input: NumericCheckInput,
    ground_truth_model_input: Optional[NumericCheckInput] = None,
    ground_truth_model_output: Optional[NumericCheckOutput] = None,
    check_settings: CheckSettings = FWD_BWD_NORMAL_CHECK,
    reset_buffer: bool = False,
    with_copy: bool = True,
) -> Tuple[Dict[str, NumericCheckOutput], Optional[CheckResult]]:
    """Check the numerical correctness of a model.

    Args:
        check_settings (CheckSettings): The check setting.
        gt_input (Any): The ground truth input.
        gt_model (Any): The ground truth model.
        actual_input (Any): The actual input.
        actual_model (Any): The actual model.
        gt_param_remap (Optional[Dict[str, str]], optional): The parameter mapping from the ground truth model to the inference
            model. Defaults to None.
        actual_params_remap (Optional[Dict[str, str]], optional): The parameter mapping from the ground truth model to the infernece
            model. Defaults to None.
        reset_buffer (bool, optional): Whether to reset the buffer of the actual model. Defaults to False.
            Set this to avoid buffer initial difference caused during previous forward, since buffer can be changed in fwd

    Returns:
        CheckResult: The check result.
    """
    # Reset buffer to avoid buffer initial difference caused during forward, since buffer can be changed in fwd
    actual_input, actual_model, actual_params_remap = actual_model_input
    comparison = {"actual": actual_model_input}
    if with_copy:
        actual_model = copy.deepcopy(actual_model)
    if ground_truth_model_input is not None:
        gt_input, gt_model, gt_param_remap = ground_truth_model_input
        if with_copy:
            gt_model = copy.deepcopy(gt_model)
        if reset_buffer:
            reset_actual_model_buffer(
                gt_model, actual_model, gt_param_remap, actual_params_remap
            )
        comparison["expected"] = ground_truth_model_input
    check_items_after_fwd = [c for c in check_settings.checklist if c.stage == "fwd"]
    check_items_after_bwd = [c for c in check_settings.checklist if c.stage == "bwd"]
    check_items_after_optimizer = [
        c for c in check_settings.checklist if c.stage == "optimizer"
    ]
    requires_optimizer = len(check_items_after_optimizer) > 0
    if not requires_optimizer:
        iteration = 1
    else:
        iteration = check_settings.iteration
    final_result = {}
    if ground_truth_model_output:
        final_result["expected"] = ground_truth_model_output
    for key, (input, model, param_remap) in comparison.items():
        # fetch the list of checkers into a dict with the name as the key and empty list as the value
        compare_list: Dict[str, Any] = {c.name: [] for c in check_settings.checklist}
        set_deterministic()
        clean_memory()

        # pyre-ignore [53]
        def extract_fn(
            check_items: List[NumericChecker],
        ) -> None:
            for check_item in check_items:
                result = check_item.extract(model, output, param_remap)
                compare_list[check_item.name].append(result)

        if requires_optimizer:
            optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        exception = None
        try:
            for i in range(iteration):
                if requires_optimizer:
                    optimizer.zero_grad()
                else:
                    model.zero_grad()
                output = model(input)
                extract_fn(check_items_after_fwd)
                if len(check_items_after_bwd) > 0:
                    try:
                        output["loss"].sum().backward()
                        extract_fn(check_items_after_bwd)
                    except Exception as ex:
                        logging.exception(ex)
                        exception = f"backward exception:{ex}"
                    if requires_optimizer:
                        try:
                            optimizer.step()
                        except Exception as ex:
                            logging.exception(ex)
                            exception = f"optimizer exception:{ex}"
                        extract_fn(check_items_after_optimizer)
                if "output" in compare_list:
                    for k, v in compare_list["output"][-1].items():
                        if k != "loss":
                            compare_list["output"][-1][k] = v.detach()
            final_result[key] = compare_list
        except Exception as ex:
            logging.exception(ex)
            exception = f"fwd exception:{ex}"
            final_result[key] = {"exception": exception}
    if "expected" in final_result:
        check_result = compare_result(
            final_result["expected"], final_result["actual"], check_settings
        )
        return (final_result, check_result)
    else:
        return (final_result, None)
