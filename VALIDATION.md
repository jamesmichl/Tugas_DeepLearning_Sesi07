# Pemeriksaan paket

Skrip modular diperiksa pada Python 3.12, PyTorch 2.5.1+cpu, Torchvision 0.20.1+cpu,
scikit-image 0.25.2, TorchMetrics 1.6.3, dan torch-fidelity 0.3.0.

Pemeriksaan yang sudah dijalankan:

- Sintaks seluruh modul dan bantuan perintah `python run.py --help`.
- Definisi arsitektur model dan fungsi difusi dibandingkan dengan AST kode notebook: sama.
- Satu epoch pada data sintetis kecil untuk AE, Sparse AE, VAE, GAN, DCGAN, DDPM, classifier, dan frozen decoder.
- Bobot serta history dapat disimpan dan dimuat kembali tanpa berubah.
- Bobot classifier tetap dibekukan saat decoder dilatih.
- Metrik rekonstruksi, transpose laten, statistik laten, dan prediksi balik DDPM menghasilkan nilai finite.
- MSE rekonstruksi identitas bernilai 0 dan SSIM bernilai 1.
- Enam PNG dapat dibuat memakai `plt`; grid loss/difusi menggunakan pembagian subplot yang konsisten.
- Mode evaluasi ulang dan plotting ulang berjalan pada checkpoint pemeriksaan.
- Split 80/10/10 dapat disimpan dan dimuat kembali; pemeriksaan split memakai dataset pengganti, bukan download ulang Fashion-MNIST.
- Gangguan FID yang disimulasikan tetap meninggalkan diagnostik MLP dan metrik parsial tersimpan.
- Jalur FID dengan Inception asli berhasil dijalankan untuk 8 input sintetis dan 8 sampel VAE yang belum terlatih.
- Notebook asli tidak diubah; checksum, enam baris metrik, dan enam gambar hasil ekstraksi diperiksa.

Pemeriksaan data kecil ini memvalidasi alur kode, bukan kualitas model. Full training 10 epoch,
FID 7.000 sampel, dan pengukuran CUDA tidak dijalankan ulang ketika paket disiapkan.
Hasil pada `results/notebook_run/` berasal dari output pengguna, dengan asal terperinci pada `ASAL_HASIL.md`.

Perubahan dari notebook terbatas pada pemisahan modul, parameter lokasi hasil,
sinkronisasi CUDA dengan argumen device, penyimpanan bertahap, dan penataan grid figur 5.
Notebook memakai seed PyTorch; skrip juga menetapkan seed NumPy dan random Python.
