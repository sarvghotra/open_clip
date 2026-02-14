exp_name="ViT-S-16"
script="./scripts/pretrain.sh"

output_dir="/network/scratch/s/sarvjeet-singh.ghotra/git/open_clip/src/logs/$exp_name"
mkdir $output_dir
cp $script $output_dir


sbatch $script $exp_name
# sbatch --array=0-2 $script $exp_name
