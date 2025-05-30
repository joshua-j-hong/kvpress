dataset="ruler"
data_dir="4096"
model="meta-llama/Meta-Llama-3.1-8B-Instruct"
# compression_ratios=(0.1 0.25 0.5)
compression_ratios=(0.1 0.25 0.5 0.75)
quanto_bits=4
press_names=("streaming_llm")

source $HOME/.zshrc
cd /ocean/projects/cis240042p/hhirairi/kvpress/evaluation/

# Check if the number of press names is less than or equal to the number of available GPUs
num_gpus=$(nvidia-smi --list-gpus | wc -l)
if [ ${#press_names[@]} -gt $num_gpus ]; then
  echo "Error: The number of press names (${#press_names[@]}) exceeds the number of available GPUs ($num_gpus)"
  exit 1
fi

# Iterate over press names and compression ratios
for i in "${!press_names[@]}"; do
  press="${press_names[$i]}"
  
  # Run each press_name on a different GPU in the background
  (
    for compression_ratio in "${compression_ratios[@]}"; do
      echo "Running press_name: $press with compression_ratio: $compression_ratio and quanto_bits: $quanto_bits on GPU cuda:$i"
      echo python evaluate.py --dataset $dataset --data_dir $data_dir --model $model --press_name $press --compression_ratio $compression_ratio --device "cuda:$i" --quanto_bits $quanto_bits
      python evaluate.py --dataset $dataset --data_dir $data_dir --model $model --press_name $press --compression_ratio $compression_ratio --device "cuda:$i" --quanto_bits $quanto_bits
    done
  ) &
done

# Wait for all background jobs to finish
wait
echo "All evaluations completed."
