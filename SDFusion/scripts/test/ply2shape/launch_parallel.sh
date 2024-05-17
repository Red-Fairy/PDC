uc_scale=3
cat=$1
path="${cat}-ply2shape-plyrot-scale3-lr0.00001"

for testset_idx in {0..2}
do
task_description="haoran0.005_gradNorm_set${testset_idx}"
for i in {0..4}
do
    start_idx=$((($i)*6))
    end_idx=$((($i+1)*6))
    bash scripts/test/ply2shape/test_guided_haoran_parallel.sh "${path}" "${i}" "250000" "${cat}" \
                            "${uc_scale}" "${start_idx}" "${end_idx}" \
                            "${testset_idx}" "${task_description}"
done
done
