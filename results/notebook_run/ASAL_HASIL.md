# Asal hasil

- Notebook: `notebooks/TugasDeepLearning2.ipynb`, salinan byte asli notebook yang sudah dijalankan.
- SHA-256 notebook: `4cd6574360036080924536f710d2dcb186a295bbbf646250df03a98085a81097`.
- `metrics.csv` dan `metrics.json`: enam baris dictionary yang tercetak pada cell index 50 notebook.
- `notebook_output.txt`: teks output tersimpan, termasuk loss training dengan presisi cetak aslinya.
- Enam PNG: diekstrak langsung dari output cell index 52, tanpa membuat sampel baru.
- `classifier_results.json`: akurasi dari notebook; dua metrik rekonstruksi MLP dari screenshot output tambahan pengguna.
- Checkpoint `.pt`, matriks bobot, statistik laten, dan histories presisi penuh tidak terdapat dalam berkas notebook. Tidak direkonstruksi atau diklaim tersedia di paket ini.
- Lingkungan notebook tercatat Python 3 dan akselerator GPU; versi paket serta jenis GPU saat training tidak terekam pada output yang tersedia.
- Isi `src/` berasal dari pemisahan kode notebook. Full training 10 epoch dan FID tidak dijalankan ulang saat penyiapan paket.
