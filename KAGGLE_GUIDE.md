# Chạy Lab Day 10 trên Kaggle

Notebook [`lab_day10.ipynb`](lab_day10.ipynb) đã có mã chạy E0, E1, E2, E3 và E4. Làm phần chuẩn bị một lần, sau đó nhấn **Run all** để chạy toàn bộ notebook theo thứ tự. E5/GRPO là phần bonus tuỳ chọn và không chạy trong luồng mặc định.

## 1. Push repo lên GitHub trước

Trong terminal của repo trên máy bạn, kiểm tra thay đổi rồi commit và push. Các lệnh sau chỉ stage các file lab; `GUIDE.md` cũng có thay đổi từ trước trong working tree, và được giữ nguyên khi tích hợp hướng dẫn Kaggle.

```bash
git status --short
git add .gitignore README.md GUIDE.md KAGGLE_GUIDE.md requirements-kaggle.txt kaggle_lab.py lab_day10.ipynb
git commit -m "feat: add Kaggle-ready Day 10 lab notebook"
git push origin main
```

Nếu `git commit` báo không có gì để commit thì kiểm tra `git status`, bảo đảm notebook và code đã được commit trước đó. Notebook Kaggle sẽ clone nhánh `main` của repo `dtc-z/K4-Track-4-Day10-VisionLanguageModel`; nếu bạn dùng fork khác, sửa `REPO_URL` ở cell đầu của notebook thành URL fork đó trước khi push.

## 2. Tạo Kaggle Notebook từ GitHub

1. Đăng nhập Kaggle, vào **Create → New Notebook**.
2. Chọn **Import Notebook** hoặc **Load from URL** trong menu File, rồi nhập URL tệp notebook trên GitHub:

   `https://github.com/dtc-z/K4-Track-4-Day10-VisionLanguageModel/blob/main/lab_day10.ipynb`

3. Mở **Notebook Settings**. Bật **Internet** và chọn **GPU** làm Accelerator. Nếu giao diện cho chọn loại GPU, T4 là lựa chọn phù hợp; notebook dùng GPU 0. Kaggle có thể cấp T4 x2 hoặc phần cứng khác tuỳ quota.
4. Mở tab **Add Input**. Tìm dataset **Road Damage Dataset - Potholes, Cracks and Manholes** của `lorenzoarcioni`, rồi bấm **Add**.
5. Bảo đảm Input có đúng một dataset chứa cả hai thư mục `images/` và `labels-YOLO/`. Notebook tự tìm các thư mục này bên dưới `/kaggle/input`.

Kaggle hỗ trợ nhập notebook từ GitHub/URL và cấu hình Internet, accelerator, cùng nguồn dữ liệu trong Notebook Editor. Xem [Kaggle Notebooks](https://www.kaggle.com/docs/notebooks) nếu tên menu trên giao diện của bạn khác.

## 3. Chạy bằng một lần Run all

Nhấn **Run all**. Không cần nhập Kaggle API key, tạo Secret, cài Kaggle CLI, mount Google Drive hay sửa cell.

Notebook tự thực hiện các bước sau:

1. Clone repo vào `/kaggle/working/K4-Track-4-Day10-VisionLanguageModel` và cấu hình tên người commit local.
2. Cài phiên bản thư viện trong `requirements-kaggle.txt`; giữ nguyên PyTorch và torchvision CUDA có sẵn trong Kaggle image.
3. Dùng dataset đã gắn trong Input, tạo split cố định theo session, kiểm tra không có session rò rỉ và hiển thị sáu ảnh nhãn mẫu.
4. Chạy E0 zero-shot, rồi E1/E2/E3 với 300 bước mỗi thí nghiệm.
5. Chạy E4 theo một biến duy nhất: rank 4 và rank 64 trên setup E1, cùng dữ liệu, seed, batch, learning rate và số bước.
6. Chấm từng thí nghiệm bằng `scripts/score.py`. Evidence được ghi vào `report/evidence/`; sáu ảnh overlay được ghi vào `report/figures/`.
7. Tạo local git commit sau E0, từng lần E1–E3 và từng biến thể E4. Cuối notebook tạo `day10-kaggle-results.bundle` để mang các commit này về repo local.

Các ô train lưu adapter cùng projector (nếu E2/E3) ở `work/checkpoints/` mỗi 100 bước và khi hoàn tất. `data/` và `work/` được gitignore; không có dữ liệu hay trọng số nào được đưa vào các commit checkpoint.

## 4. Sau khi notebook chạy xong

Đọc kết quả `report/evidence/*.json` và điền `REPORT.md` bằng số liệu thật của bạn. Notebook không tự điền phần giải thích cá nhân trong 10 câu hỏi.

Các commit được tạo trong bản clone ở Kaggle; chúng chưa được push lên GitHub. Tải git bundle từ liên kết `day10-kaggle-results.bundle` ở cell cuối khi runtime còn mở. Nếu muốn giữ output trong Kaggle sau khi đóng runtime, lưu một notebook version và tải bundle từ mục Output.

Trên máy local, trong repo mà bạn đã push trước khi chạy Kaggle, nhập các commit checkpoint và push evidence/báo cáo:

```bash
git fetch /path/to/day10-kaggle-results.bundle refs/heads/main:refs/remotes/kaggle/main
git merge --ff-only refs/remotes/kaggle/main

# Điền REPORT.md rồi commit báo cáo
git add REPORT.md
git commit -m "report: add Day 10 experiment results"
git push origin main
```

Thay `/path/to/day10-kaggle-results.bundle` bằng đường dẫn file bạn vừa tải. Nếu đã thay đổi local branch sau lần push ban đầu, hãy tích hợp các commit đó trước khi chạy `merge --ff-only`.

Cuối cùng, chạy `python scripts/check_submission.py` trong repo local sau khi đã tải đủ evidence và điền `REPORT.md`. Không commit `data/`, `work/`, `.env`, `*.pt`, `*.bin` hoặc `*.safetensors`.

## 5. Các giới hạn thực tế

- E0–E4 gồm năm lượt huấn luyện và năm lượt inference trên test; thời gian tuỳ GPU Kaggle, tải model và quota. Hãy để ý giới hạn thời lượng session của Kaggle.
- Model Qwen3-VL-4B tải từ Hugging Face; Kaggle Internet phải bật trong lúc chạy.
- Nếu dataset không được tìm thấy, xác nhận đã bấm **Add** và thư mục nguồn thực sự có `images/` cùng `labels-YOLO/`. Nếu notebook báo tìm thấy nhiều bộ dataset, gỡ các Input dataset trùng.
- Nếu thiếu VRAM, xem bảng **Khi gặp sự cố** trong `GUIDE.md`. Thay đổi batch/gradient accumulation cần được ghi rõ trong báo cáo vì sẽ khác cấu hình chuẩn.
- Kaggle runtime có thể ngắt hoặc hết quota. Checkpoint adapter trong `work/checkpoints/` hữu ích cho phiên đang chạy, nhưng notebook hiện không tự tiếp tục từ bước train dang dở sau khi mở runtime mới.
