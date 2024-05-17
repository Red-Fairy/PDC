for testset_idx in {1..2}
do
    for i in {0..7}
    do
        start_idx=$((($i)*8))
        end_idx=$((($i+1)*8))
        bash scripts/test/ply2shape/test_ply2shape_guided_haoran_parallel.sh "slider_drawer-ply2shape-plyrot-scale3-lr0.00001" "${i}" "250000" "${start_idx}" "${end_idx}"
    done
done