#!/bin/bash

echo "Çalışan driverlar ve sıcaklıkları..."
nvidia-smi --query-gpu=gpu_name,temperature.gpu --format=csv


echo "Top komutu Kullanılan Cpu Ve Mib Mem"
top -bn1 | grep "Cpu(s)" && top -bn1 | grep "MiB Mem"

echo "Top komutu Kullanılan memory alanı ve Cuda Versiyonu :"
top -bn1 | grep "MiB Swap" && nvidia-smi | grep "CUDA Version"

echo " Kullanılan GPU Mevcutsa : Process_ID, Process_Name , Used_GPU_Memory  :"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv

echo "Gpu Utilizasyonu ve Kullanılan Memory : "
paste <(nvidia-smi --query-gpu=utilization.gpu --format=csv) <(nvidia-smi --query-gpu=memory.used --format=csv)

echo "Kaç Watt kullanılıyor ve Grafik Kartı Saniye Başına Atımı "
paste <(nvidia-smi --query-gpu=power.draw --format=csv) <(nvidia-smi --query-gpu=clocks.gr --format=csv)
