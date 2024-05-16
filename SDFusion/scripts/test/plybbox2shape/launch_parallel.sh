# for ply_scale in {0,1,2,3}
# do
# for bbox_scale in {0,1,2,3}
# do
uc_scale=3
for i in {0..7}
do
    start_idx=$((($i)*5))
    end_idx=$((($i+1)*5))
    bash scripts/test/plybbox2shape/test_guided_haoran_parallel.sh "slider_drawer-plybbox2shape-plyrot-lr0.00001" "${i}" "200000" "${uc_scale}" "${start_idx}" "${end_idx}"
done
# sleep 180
# done
# done