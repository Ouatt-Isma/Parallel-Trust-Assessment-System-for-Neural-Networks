import os

folder = "good_plots/MNIST"
output_file = "filelistmnist.tex"

files = []

# Step 1: Gather all file paths
for epsval in sorted(os.listdir(folder)):
    epsval_path = os.path.join(folder, epsval)
    if not os.path.isdir(epsval_path):
        continue
    for setup in sorted(os.listdir(epsval_path)):
        setup_path = os.path.join(epsval_path, setup)
        filename = os.path.join(setup_path, "all.pdf")
        if os.path.isfile(filename):
            files.append(filename.replace("\\", "/"))  # Ensure LaTeX-compatible paths

# Step 2: Write LaTeX code with subfigures using \subfloat in 3x3 grid
with open(output_file, "w") as f:
    for i, filepath in enumerate(files):
        if i % 9 == 0:
            f.write("\\begin{figure}[htbp]\n\\centering\n")

        sp = filepath.split("/")[-2]
        caption = sp.split("y")[0] + "+y" + sp.split("y")[1]

        f.write(f"\\subfloat[{caption}]{{%\n")
        f.write(f"  \\includegraphics[width=0.3\\textwidth]{{{filepath}}}%\n")
        f.write("}\n")

        if (i + 1) % 3 == 0:
            f.write("\\\\\n")  # New row after 3 images

        if (i + 1) % 9 == 0 or (i + 1) == len(files):
            epsv = filepath.split("/")[-3].split("=")[1]
            f.write(f"\\caption{{epsilon={epsv}}}\n\\end{{figure}}\n\n")
