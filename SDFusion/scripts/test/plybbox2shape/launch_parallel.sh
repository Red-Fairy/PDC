for i in {2..7}
do
    start_idx=$((($i-2)*6))
    end_idx=$(($i*6))
    bash scripts/test/plybbox2shape/test_guided_haoran_parallel.sh "slider_drawer-plybbox2shape-plyrot-lr0.00001" "${i}" "150000" "0" "${start_idx}" "${end_idx}"
done