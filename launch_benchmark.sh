#!/bin/bash
set -xe
# model card
# imagenet: https://github.com/mengfei25/pytorch-examples/tree/develop/imagenet
# dcgan: https://github.com/mengfei25/pytorch-examples/tree/develop/dcgan
export TORCHINDUCTOR_FREEZING=1
export LD_PRELOAD=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib/libiomp5.so:${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib/libjemalloc.so
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:-1,muzzy_decay_ms:-1"
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=1
function main {
    # set common info
    source oob-common/common.sh
    init_params $@
    fetch_device_info
    set_environment

    # if multiple use 'xxx,xxx,xxx'
    model_name_list=($(echo "${model_name}" |sed 's/,/ /g'))
    batch_size_list=($(echo "${batch_size}" |sed 's/,/ /g'))

    # generate benchmark
    for model_name in ${model_name_list[@]}
    do
        # cache
        python benchmarks/dynamo/${MODEL_SUITE}.py --performance --inference --$precision -dcpu -n50 \
            --num_iter 3 --num_warmup 1 \
            --no-skip --dashboard --only $model_name --timeout 9000 --backend=inductor --freezing \
            ${addtion_options} || true
        #
        # for batch_size in ${batch_size_list[@]}
        # do
        # clean workspace
        logs_path_clean
        # generate launch script for multiple instance
        if [ "${OOB_USE_LAUNCHER}" == "1" ] && [ "${device}" != "cuda" ];then
            generate_core_launcher
        else
            generate_core
        fi
        # launch
        echo -e "\n\n\n\n Running..."
        # cat ${excute_cmd_file} |column -t > ${excute_cmd_file}.tmp
        cat ${excute_cmd_file}  > ${excute_cmd_file}.tmp
        mv ${excute_cmd_file}.tmp ${excute_cmd_file}
        source ${excute_cmd_file}
        echo -e "Finished.\n\n\n\n"
        # collect launch result
        collect_perf_logs
        # done
    done
}

# run
function generate_core {
    # generate multiple instance script
    for(( i=0; i<instance; i++ ))
    do
        real_cores_per_instance=$(echo ${device_array[i]} |awk -F, '{print NF}')
        log_file="${log_dir}/rcpi${real_cores_per_instance}-ins${i}.log"

        # instances
        if [ "${device}" != "cuda" ];then
            OOB_EXEC_HEADER=" numactl -m $(echo ${device_array[i]} |awk -F ';' '{print $2}') "
            OOB_EXEC_HEADER+=" -C $(echo ${device_array[i]} |awk -F ';' '{print $1}') "
        else
            OOB_EXEC_HEADER=" CUDA_VISIBLE_DEVICES=${device_array[i]} "
        fi
        printf " ${OOB_EXEC_HEADER} \
            python benchmarks/dynamo/${MODEL_SUITE}.py --performance --inference --$precision -dcpu -n50 \
            --num_iter $num_iter --num_warmup $num_warmup \
            --no-skip --dashboard --only $model_name --timeout 9000 --backend=inductor --freezing \
            ${addtion_options} \
        > ${log_file} 2>&1 &  \n" |tee -a ${excute_cmd_file}
        if [ "${numa_nodes_use}" == "0" ];then
            break
        fi
    done
    echo -e "\n wait" >> ${excute_cmd_file}
}

function generate_core_launcher {
    # generate multiple instance script
    for(( i=0; i<instance; i++ ))
    do
        real_cores_per_instance=$(echo ${device_array[i]} |awk -F, '{print NF}')
        log_file="${log_dir}/rcpi${real_cores_per_instance}-ins${i}.log"

        printf "python -m oob-common.launch --enable_jemalloc \
                    --core_list $(echo ${device_array[@]} |sed 's/;.//g') \
                    --log_file_prefix rcpi${real_cores_per_instance} \
                    --log_path ${log_dir} \
                    --ninstances ${#device_array[@]} \
                    --ncore_per_instance ${real_cores_per_instance} \
            benchmarks/dynamo/${MODEL_SUITE}.py --performance --inference --$precision -dcpu -n50 \
                --num_iter $num_iter --num_warmup $num_warmup \
                --no-skip --dashboard --only $model_name --timeout 9000 --backend=inductor --freezing \
                ${addtion_options} \
        > /dev/null 2>&1 &  \n" |tee -a ${excute_cmd_file}
        break
    done
    echo -e "\n wait" >> ${excute_cmd_file}
}

# download common files
rm -rf oob-common && git clone https://github.com/zxd1997066/oob-common.git

# Start
main "$@"
