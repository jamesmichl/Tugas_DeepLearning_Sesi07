import argparse
import matplotlib

matplotlib.use("Agg")

from src.experiment import run_experiment


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eksperimen generative models Fashion-MNIST")
    parser.add_argument("--output-dir", default="results/new_run")
    parser.add_argument("--data-dir", default="data")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--evaluate-only", action="store_true", help="Evaluasi bobot tersimpan tanpa training")
    mode.add_argument("--figures-only", action="store_true", help="Buat gambar dari bobot dan evaluasi tersimpan")
    parser.add_argument("--no-fid", action="store_true", help="Lewati FID; hasil evaluasi belum lengkap")
    args = parser.parse_args()
    run_experiment(args.output_dir, args.data_dir, args.evaluate_only, args.figures_only, not args.no_fid)
