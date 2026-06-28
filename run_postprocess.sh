 
PROC=./process_saw_xy_vortices
BASE=out/PT_main_XY_SI_2D_3
OUT=out/PT_main_XY_SI_2D_3_twist

mkdir -p "$OUT"

sbatch --time=03:30:10 -A proj_1722 --wrap="\"$PROC\" --angles \"$BASE/angles_3600_1776716185.out\" --dirs \"$BASE/dirs_3600_1776716185.out\" --L 3600 --dim 2 --side 3424 --outdir \"$OUT/\" --chain -1 --angle-order ring --twist-mode contacts"

sbatch --time=03:30:10 -A proj_1722 --wrap="\"$PROC\" --angles \"$BASE/angles_4900_1776716308.out\" --dirs \"$BASE/dirs_4900_1776716308.out\" --L 4900 --dim 2 --side 4659 --outdir \"$OUT/\" --chain -1 --angle-order ring --twist-mode contacts"

sbatch --time=03:30:10 -A proj_1722 --wrap="\"$PROC\" --angles \"$BASE/angles_2500_1776716305.out\" --dirs \"$BASE/dirs_2500_1776716305.out\" --L 2500 --dim 2 --side 2379 --outdir \"$OUT/\" --chain -1 --angle-order ring --twist-mode contacts"


# "$PROC" --angles "$BASE/angles_3600_1776628882.out" --dirs "$BASE/dirs_3600_1776628882.out" --L 3600 --dim 2 --side 3424 --outdir "$OUT/" --chain -1   #--threads 8
# "$PROC" --angles "$BASE/angles_4900_1776716308.out" --dirs "$BASE/dirs_4900_1776716308.out" --L 4900 --dim 2 --side 4659 --outdir "$OUT/" --chain -1   #--threads 8
# "$PROC" --angles "$BASE/angles_2500_1776716305.out" --dirs "$BASE/dirs_2500_1776716305.out" --L 2500 --dim 2 --side 2379 --outdir "$OUT/" --chain -1   #--threads 8



# sbatch --time=03:30:10 -A proj_1722 --wrap="\"$PROC\" --angles \"$BASE/angles_1000_1772395548.out\" --dirs \"$BASE/dirs_1000_1772395548.out\" --L 1000 --dim 2 --side 954 --outdir \"$OUT/\" --chain -1 --angle-order ring --twist-mode contacts"

# sbatch --time=03:30:10 -A proj_1722 --wrap="\"$PROC\" --angles \"$BASE/angles_1600_1772395548.out\" --dirs \"$BASE/dirs_1600_1772395548.out\" --L 1600 --dim 2 --side 1524 --outdir \"$OUT/\" --chain -1 --angle-order ring --twist-mode contacts"

# sbatch --time=03:30:10 -A proj_1722 --wrap="\"$PROC\" --angles \"$BASE/angles_2500_1772395548.out\" --dirs \"$BASE/dirs_2500_1772395548.out\" --L 2500 --dim 2 --side 2379 --outdir \"$OUT/\" --chain -1 --angle-order ring --twist-mode contacts"

# sbatch --time=03:30:10 -A proj_1722 --wrap="\"$PROC\" --angles \"$BASE/angles_3600_1772395549.out\" --dirs \"$BASE/dirs_3600_1772395549.out\" --L 3600 --dim 2 --side 3424 --outdir \"$OUT/\" --chain -1 --angle-order ring --twist-mode contacts"

# sbatch --time=03:30:10 -A proj_1722 --wrap="\"$PROC\" --angles \"$BASE/angles_4900_1772395551.out\" --dirs \"$BASE/dirs_4900_1772395551.out\" --L 4900 --dim 2 --side 4659 --outdir \"$OUT/\" --chain -1 --angle-order ring --twist-mode contacts"




# sbatch --time=03:30:10 -A proj_1722 --wrap="\"$PROC\" --angles \"$BASE/angles_900_1774996722.out\" --dirs \"$BASE/dirs_900_1774996722.out\" --L 900 --dim 2 --side 859 --outdir \"$OUT/\" --chain -1 --angle-order ring --twist-mode contacts"

# sbatch --time=03:30:10 -A proj_1722 --wrap="\"$PROC\" --angles \"$BASE/angles_1000_1775005313.out\" --dirs \"$BASE/dirs_1000_1775005313.out\" --L 1000 --dim 2 --side 954 --outdir \"$OUT/\" --chain -1 --angle-order ring --twist-mode contacts"

# sbatch --time=03:30:10 -A proj_1722 --wrap="\"$PROC\" --angles \"$BASE/angles_2500_1775003986.out\" --dirs \"$BASE/dirs_2500_1775003986.out\" --L 2500 --dim 2 --side 2379 --outdir \"$OUT/\" --chain -1 --angle-order ring --twist-mode contacts"

# sbatch --time=03:30:10 -A proj_1722 --wrap="\"$PROC\" --angles \"$BASE/angles_3600_1775004712.out\" --dirs \"$BASE/dirs_3600_1775004712.out\" --L 3600 --dim 2 --side 3424 --outdir \"$OUT/\" --chain -1 --angle-order ring --twist-mode contacts"

# sbatch --time=03:30:10 -A proj_1722 --wrap="\"$PROC\" --angles \"$BASE/angles_4900_1775004862.out\" --dirs \"$BASE/dirs_4900_1775004862.out\" --L 4900 --dim 2 --side 4659 --outdir \"$OUT/\" --chain -1 --angle-order ring --twist-mode contacts "


# sbatch --time=03:30:10 -A proj_1722 --wrap="\"$PROC\" --angles \"$BASE/angles_300_1774169921.out\" --dirs \"$BASE/dirs_300_1774169921.out\" --L 300 --dim 2 --side 289 --outdir \"$OUT/\" --chain -1 --angle-order ring --twist-mode contacts"

# sbatch --time=03:30:10 -A proj_1722 --wrap="\"$PROC\" --angles \"$BASE/angles_500_1774169921.out\" --dirs \"$BASE/dirs_500_1774169921.out\" --L 500 --dim 2 --side 479 --outdir \"$OUT/\" --chain -1 --angle-order ring --twist-mode contacts"

# sbatch --time=03:30:10 -A proj_1722 --wrap="\"$PROC\" --angles \"$BASE/angles_750_1774171273.out\" --dirs \"$BASE/dirs_750_1774171273.out\" --L 750 --dim 2 --side 716 --outdir \"$OUT/\" --chain -1 --angle-order ring --twist-mode contacts"

# sbatch --time=03:30:10 -A proj_1722 --wrap="\"$PROC\" --angles \"$BASE/angles_900_1774171273.out\" --dirs \"$BASE/dirs_900_1774171273.out\" --L 900 --dim 2 --side 859 --outdir \"$OUT/\" --chain -1 --angle-order ring --twist-mode contacts"

# sbatch --time=03:30:10 -A proj_1722 --wrap="\"$PROC\" --angles \"$BASE/angles_1600_1774256268.out\" --dirs \"$BASE/dirs_1600_1774256268.out\" --L 1600 --dim 2 --side 1524 --outdir \"$OUT/\" --chain -1 --angle-order ring --twist-mode contacts"

# sbatch --time=03:30:10 -A proj_1722 --wrap="\"$PROC\" --angles \"$BASE/angles_2500_1774201369.out\" --dirs \"$BASE/dirs_2500_1774201369.out\" --L 2500 --dim 2 --side 2379 --outdir \"$OUT/\" --chain -1 --angle-order ring --twist-mode contacts"

# sbatch --time=03:30:10 -A proj_1722 --wrap="\"$PROC\" --angles \"$BASE/angles_3600_1774211749.out\" --dirs \"$BASE/dirs_3600_1774211749.out\" --L 3600 --dim 2 --side 3424 --outdir \"$OUT/\" --chain -1 --angle-order ring --twist-mode contacts"

# sbatch --time=03:30:10 -A proj_1722 --wrap="\"$PROC\" --angles \"$BASE/angles_1000_1774725478.out\" --dirs \"$BASE/dirs_1000_1774725478.out\" --L 1000 --dim 2 --side 954 --outdir \"$OUT/\" --chain -1 --angle-order ring --twist-mode contacts"