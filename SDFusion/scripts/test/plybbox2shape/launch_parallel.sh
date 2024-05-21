uc_scale=3
cat=$1
path="${cat}-plybbox2shape-plyrot-lr0.00001"

for testset_idx in {0..0}
do
task_description="haoran0.005_gradNorm_set${testset_idx}"
for i in {0..7}
do
    start_idx=$((($i)*63))
    end_idx=$((($i+1)*63))
    bash scripts/test/plybbox2shape/test_guided_haoran_parallel.sh "${path}" "${i}" "200000" "${cat}" \
                            "${uc_scale}" "${start_idx}" "${end_idx}" \
                            "${testset_idx}" "${task_description}"
done
done
