#!/bin/sh
cd /workspace
while [ ! -f results/reproduccion/COMPLETED.json ]; do
 if [ -f results/reproduction_exit.txt ] && [ "$(cat results/reproduction_exit.txt)" != 0 ]; then exit 1; fi
 sleep 10
done
python tests/test_models.py >> results/unit_tests_final.log 2>&1 || exit 1
python scripts/ablaciones.py >> results/ablations.log 2>&1 || { echo 1 > results/ablations_exit.txt; exit 1; }
echo 0 > results/ablations_exit.txt
python scripts/validar_y_redactar.py > results/report.log 2>&1
status=$?
echo "$status" > results/report_exit.txt
if [ "$status" = 0 ]; then touch results/STOP_MONITOR; fi
exit "$status"
