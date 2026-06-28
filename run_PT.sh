sbatch --time=15-10:0 --gpus=1 -A proj_1722 --wrap="./mc_modeling 700 0.258390 0.2638 out/PT_main"
sbatch --time=15-10:0 --gpus=1 -A proj_1722 --wrap="./mc_modeling 650 0.25490 0.2640 out/PT_main"
sbatch --time=15-10:0 --gpus=1 -A proj_1722 --wrap="./mc_modeling 600 0.258592 0.2642 out/PT_main"
sbatch --time=15-10:0 --gpus=1 -A proj_1722 --wrap="./mc_modeling 550 0.2595 0.2645 out/PT_main"

sbatch --time=15-10:0 --gpus=1 -A proj_1722 --wrap="./mc_modeling 500 0.2600 0.2648 out/PT_main"
sbatch --time=14-10:0 --gpus=1 -A proj_1722 --wrap="./mc_modeling 450 0.2605 0.2655 out/PT_main"
sbatch --time=12-10:0 --gpus=1 -A proj_1722 --wrap="./mc_modeling 400 0.2615 0.2665 out/PT_main"
sbatch --time=11-10:0 --gpus=1 -A proj_1722 --wrap="./mc_modeling 350 0.2645 0.2698 out/PT_main"
sbatch --time=8-10:0 --gpus=1 -A proj_1722 --wrap="./mc_modeling 300 0.2735 0.2790 out/PT_main"

sbatch --time=1-10:0 --gpus=1 -A proj_1722 --wrap="./mc_modeling 100 0.2735 0.3872790 out/PT_main"