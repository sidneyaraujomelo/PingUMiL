source experiment_config.sh
MODELS=("graphsage_maxpool" "graphsage_seq" "graphsage_meanpool" "graphsage_mean" "gcn")
#MODELS=("graphsage_seq")

for MODEL in "${MODELS[@]}";
do
	echo "Modelo : ${MODEL}"
	OUTPUT_FOLDER="output_${TEST_FOLDER}_${MODEL}"
	python exp_scripts/edge_classifier_training.py prov $TEST_FOLDER/ ${OUTPUT_FOLDER}/unsup-${TEST_FOLDER}/${MODEL}_small_0.000010 val ${TEST_FOLDER}/clf
done
