# Hướng dẫn chạy AuraNet trên Kaggle

Tài liệu này hướng dẫn cách thiết lập và chạy mô hình AuraNet trên môi trường Kaggle Notebooks.

## Chuẩn bị

### Bước 1: Upload dữ liệu và mã nguồn
1. Tạo một Kaggle dataset chứa:
   - Mã nguồn AuraNet (tất cả các file .py)
   - File cấu hình (config_celeb_df_memory_optimized.yaml)
   - Pretrained weights (convnextv2_pico_1k_224_fcmae.pt)

2. Tạo một Kaggle dataset khác chứa:
   - Dữ liệu Celeb-DF (trong cùng cấu trúc thư mục như trong dự án gốc)

### Bước 2: Thiết lập Kaggle Notebook
1. Tạo một Kaggle Notebook mới
2. Trong phần "Add data", thêm dataset chứa mã nguồn AuraNet và dataset dữ liệu Celeb-DF
3. Thiết lập Accelerator là GPU (P100)

## Chạy training

### Cách 1: Sử dụng script bash
```bash
# Cấp quyền thực thi cho script
chmod +x /kaggle/input/auranet/kaggle_train.sh

# Chạy script
./kaggle/input/auranet/kaggle_train.sh
```

### Cách 2: Chạy trực tiếp với Python
```python
import os
import sys

# Thêm đường dẫn mã nguồn vào Python path
sys.path.append('/kaggle/working/AuraNet')
sys.path.append('/kaggle/working/AuraNet/src')

# Sao chép file từ input đến working directory
!mkdir -p /kaggle/working/AuraNet
!cp -r /kaggle/input/auranet/* /kaggle/working/AuraNet/

# Chạy training
!python /kaggle/working/AuraNet/train_celeb_df.py \
    --config /kaggle/working/AuraNet/config_celeb_df_memory_optimized.yaml \
    --mode pretrain \
    --data_root /kaggle/input \
    --gpus 2 \
    --use_pretrained yes \
    --pretrained_path /kaggle/input/convnextv2-pico/pytorch/default/1/convnextv2_pico_1k_224_fcmae.pt \
    --kaggle \
    --kaggle_working_dir /kaggle/working/AuraNet
```

## Cấu hình thông số

Bạn có thể điều chỉnh các thông số sau trong script hoặc lệnh chạy:

- `--mode`: Chế độ training (pretrain, finetune, both)
- `--gpus`: Số lượng GPU sử dụng (1-2)
- `--use_pretrained`: Có sử dụng pretrained weights hay không (yes/no)
- `--batch_size`: Kích thước batch
- `--epochs`: Số epoch
- `--learning_rate`: Tốc độ học

## Xử lý lỗi thường gặp

1. **Lỗi "No such file or directory"**:
   - Kiểm tra đường dẫn file trong lệnh chạy
   - Đảm bảo đã sao chép tất cả file cần thiết vào /kaggle/working

2. **Lỗi OutOfMemory**:
   - Giảm batch_size trong file config
   - Sử dụng config_celeb_df_memory_optimized.yaml
   - Thêm tham số `--memory_optimization` khi chạy

3. **Lỗi import module**:
   - Đảm bảo đã thêm đường dẫn vào sys.path
   - Kiểm tra cấu trúc thư mục trong /kaggle/working

## Tối ưu hóa

Để tối ưu hóa quá trình training trên Kaggle:

1. Sử dụng memory-optimized config
2. Giảm kích thước ảnh xuống 128x128 để tiết kiệm bộ nhớ
3. Lưu checkpoint thường xuyên để có thể tiếp tục training sau khi runtime kết thúc
4. Sử dụng mixed precision training

## Lưu kết quả

Kết quả training sẽ được lưu trong thư mục `/kaggle/working/AuraNet/logs/`. Đảm bảo xuất file logs ra ngoài trước khi kết thúc Notebook để không bị mất dữ liệu.
