# XGB Recommender Pipeline

File chính: `src/xgb_recommender.py`

Đây là pipeline end-to-end để huấn luyện mô hình recommendation theo tháng, tạo candidate items cho từng user, train mô hình ranking/classification, đánh giá validation và xuất file submission.

## Luồng tổng quát

1. Parse cấu hình từ command line.
2. Đọc dữ liệu transaction và metadata item.
3. Xác định các tháng dùng để train.
4. Build training dataset cho từng tháng train.
5. Build validation dataset.
6. Train model lần đầu với validation.
7. Đánh giá validation month.
8. Refit model cuối cùng trên train + validation.
9. Predict cho test month.
10. Tùy chọn evaluate submission với ground truth.

## Dữ liệu đầu vào

Mặc định pipeline dùng:

- `data/transaction_full_2025.parquet`: dữ liệu giao dịch.
- `data/items.parquet`: metadata item, nếu tồn tại.
- `ground_truth.json`: ground truth dùng cho evaluation tùy chọn.

Nếu không tìm thấy metadata item, pipeline vẫn chạy nhưng bỏ qua item features.

## Tháng train, validation và test

Các tham số mặc định:

- `--val-month 2025-10`
- `--test-month 2025-11`
- `--train-months`: nếu không truyền, pipeline lấy 2 tháng trước validation month.

Ví dụ với mặc định `val-month = 2025-10`, train months sẽ là:

- `2025-08`
- `2025-09`

Có thể tự chỉ định:

```bash
python src/xgb_recommender.py --train-months 2025-06,2025-07,2025-08
```

## Candidate generation và feature engineering

Pipeline tạo candidates dựa trên nhiều nguồn:

- Items user từng tương tác trong lịch sử.
- Items phổ biến toàn cục.
- Items phổ biến theo location.
- Items phổ biến theo category gần đây của user.
- Items phổ biến theo brand gần đây của user.
- Items co-buy từ các item user đã mua/tương tác.

Các tham số chính:

- `--history-days`: số ngày lịch sử dùng để tạo feature, mặc định `270`.
- `--recent-days`: cửa sổ recent behavior, mặc định `90`.
- `--candidate-top`: số candidate chính, mặc định `80`.
- `--popular-top`: số item phổ biến toàn cục, mặc định `60`.
- `--location-top`: số item theo location, mặc định `50`.
- `--category-top`: số item theo category, mặc định `20`.
- `--brand-top`: số item theo brand, mặc định `20`.
- `--cobuy-top`: số item co-buy mỗi anchor item, mặc định `20`.

Candidate cache được lưu mặc định ở:

```text
cache/candidates
```

Có thể tắt cache bằng:

```bash
python src/xgb_recommender.py --no-candidate-cache
```

## Build training dataset

Với mỗi train month, pipeline gọi:

```python
build_training_dataset_for_month(...)
```

Kết quả các tháng được ghép lại thành `train_df`.

Training dataset có label:

- `1`: user có tương tác/mua item trong label month.
- `0`: negative candidate.

Negative sampling được điều khiển bằng:

```bash
--negative-ratio 6.0
```

Validation dataset được build riêng cho `--val-month` với `downsample=False` để đánh giá ranking đầy đủ hơn.

## Model training

Pipeline hỗ trợ các backend:

- `xgboost`
- `lightgbm`
- `linear_regression`

Mặc định:

```bash
--model xgboost
```

Với tree models, objective có thể là:

- `rank_ndcg`
- `rank_pairwise`
- `binary`

Mặc định:

```bash
--xgb-objective rank_ndcg
```

Ranking mode dùng group theo `customer_id` để học thứ tự candidate trong từng user.

Các tham số model chính:

- `--n-estimators 500`
- `--early-stopping-rounds 50`
- `--n-jobs 0`
- `--xgb-device cpu`

## Validation

Sau khi train model lần đầu, pipeline chạy:

```python
evaluate_month_chunked(...)
```

Validation prediction được lưu thành:

```text
validation_{val_month}_submission.json
```

Ví dụ:

```text
validation_2025-10_submission.json
```

## Refit final model

Mặc định pipeline refit model cuối cùng trên:

```text
train_df + val_df
```

Số cây dùng cho final model được lấy từ best iteration của model validation nếu có early stopping.

Nếu không muốn refit:

```bash
python src/xgb_recommender.py --no-refit-on-validation
```

## Prediction

Pipeline predict test month bằng:

```python
predict_month_chunked(...)
```

Output mặc định:

```text
submission_xgb.json
```

Có thể đổi file output:

```bash
python src/xgb_recommender.py --submission my_submission.json
```

Mặc định pipeline predict cho users liên quan tới test month. Có thể predict cho toàn bộ users có lịch sử trước test month bằng:

```bash
python src/xgb_recommender.py --predict-all-history-users
```

## Evaluation tùy chọn

Đánh giá test month dựa trên transaction parquet:

```bash
python src/xgb_recommender.py --evaluate-test-month
```

Đánh giá submission với `ground_truth.json`:

```bash
python src/xgb_recommender.py --eval-ground-truth
```

Chỉ nên dùng `--eval-ground-truth` khi ground truth thật sự khớp với `--test-month`.

## Chế độ predict tháng 2026-01

Flag đặc biệt:

```bash
python src/xgb_recommender.py --predict-jan-2026
```

Khi bật flag này, pipeline tự cấu hình:

- `test_month = 2026-01`
- `val_month = 2025-12`
- train months là `2025-01` đến `2025-11`
- refit thêm validation month `2025-12`
- predict cho toàn bộ users có lịch sử trước `2026-01-01`
- output mặc định là `submission_2026-01.json`

Nếu truyền `--train-months` cùng lúc, tham số đó sẽ bị bỏ qua trong mode này.

## Ví dụ chạy

Chạy mặc định:

```bash
python src/xgb_recommender.py
```

Chạy với LightGBM:

```bash
python src/xgb_recommender.py --model lightgbm
```

Chạy XGBoost ranking trên CUDA:

```bash
python src/xgb_recommender.py --model xgboost --xgb-device cuda
```

Train nhiều tháng hơn và xuất submission riêng:

```bash
python src/xgb_recommender.py ^
  --train-months 2025-06,2025-07,2025-08,2025-09 ^
  --val-month 2025-10 ^
  --test-month 2025-11 ^
  --submission submission_2025-11.json
```

## Tóm tắt ngắn

Pipeline hiện tại:

```text
transactions/items
    -> candidate generation + feature engineering theo tháng
    -> train_df và val_df
    -> train model với validation
    -> evaluate validation
    -> refit trên train + validation
    -> predict test month
    -> submission JSON
```
