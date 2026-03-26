
# BÁO CÁO BÀI TẬP: DEEP LEARNING AND IT'S APPLICATIONS: PHÂN LOẠI ẢNH TRÊN TẬP DỮ LIỆU CIFAR-10

---

- Giảng viên hướng dẫn: TS. Lê Thành Sách

- Group_5 - Thành viên:

    * Hà Thanh Bình -2470732
    
    * Trần Đăng Hùng -2470750

    *Nguyễn Võ Thái Triều-2470577
---
---

## 1. Giới thiệu
Excercise này thực hiện bài toán phân loại ảnh trên tập dữ liệu CIFAR-10, nhằm so sánh hiệu năng của nhiều mô hình Deep Learning khác nhau:
- Softmax Regression
- MLP
- CNN
- Vision Transformer (ViT)
- Custom Transformer
- CNN + Transformer
- RNN (LSTM/GRU)

## 2. Dataset
- CIFAR-10
- 50,000 ảnh train, 10,000 ảnh test
- Chia: 45.000 train / 5.000 validation
- Augmentation: RandomCrop, HorizontalFlip
- Normalize theo mean/std CIFAR-10

## 3. Cấu trúc thư mục

```
project/
├── Exercise_0.ipynb
├── models/
├── utils/
├── results/
├── README.md
```

## 4. Cài đặt môi trường

```bash
pip install torch torchvision matplotlib numpy
```

## 5. Training Pipeline
- Optimizer: Adam
- Batch size: 128
- Epochs: 30
- Loss: CrossEntropyLoss
- Có validation mỗi epoch
- Lưu best checkpoint theo validation accuracy

## 6. Các mô hình

### Baseline
- Softmax Regression
- MLP
- CNN
- ViT (built-in)

### Advanced
- Custom Transformer Encoder
- Custom ViT
- CNN + Transformer
- Overlap Patch ViT
- Channel-as-Token

### RNN
- LSTM (row-wise)
- GRU (row-wise)
- LSTM (patch-wise)

## 7. Kết quả chính

- CNN và CNN+Transformer đạt kết quả tốt nhất
- ViT chưa vượt CNN do dữ liệu nhỏ
- RNN không phù hợp cho dữ liệu ảnh

## 8. Nhận xét

- CNN khai thác tốt cấu trúc không gian
- Transformer mạnh về quan hệ toàn cục
- Hybrid model (CNN + Transformer) hiệu quả nhất

## 9. Hạn chế

- Không sử dụng pretraining
- Dataset nhỏ

## 10. Hướng phát triển

- Sử dụng pretrained ViT
- Tăng dữ liệu / augmentation
- Tuning learning rate, architecture
- Thêm F1-score

