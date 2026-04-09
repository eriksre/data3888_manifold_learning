# V2 Reconstruction

Edit the config in `recover_stock_price_v2.py`, then run:

```bash
python recover_stock_price_v2.py
```

Use:
- `MODE = "single"` with `INPUT_CSV` and `OUTPUT_CSV`
- `MODE = "folder"` with `BOOK_DIR` and `OUTPUT_DIR`

The output CSV keeps the original columns and adds `global_second`, `recovered_rank`, `recovered_price`, and `base_price`.
