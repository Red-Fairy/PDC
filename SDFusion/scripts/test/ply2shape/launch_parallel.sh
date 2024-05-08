for i in {2..8}
do
    start_idx=$((($i-2)*6))
    end_idx=$(($i*6))
    bash scripts/test/ply2shape/test_ply2shape_guided_haoran_parallel.sh "slider_drawer-ply2shape-plyrot-scale3-lr0.00001" "${i}" "250000" "0" "${start_idx}" "${end_idx}"
done