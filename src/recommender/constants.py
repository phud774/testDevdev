from __future__ import annotations


BILL_COL = "bill_id"
USER_COL = "customer_id"
ITEM_COL = "item_id"
DATE_COL = "updated_date"
PRICE_COL = "price"
DISCOUNT_COL = "discount"
QTY_COL = "quantity"
LOC_COL = "location"

CAT_L1_COL = "category_l1"
CAT_L2_COL = "category_l2"
CAT_L3_COL = "category_l3"
CAT_COL = "category"
BRAND_COL = "brand"
SALE_STATUS_COL = "sale_status"
ITEM_CATALOG_PRICE_COL = "item_catalog_price"
ITEM_DESC_LEN_COL = "item_description_len"


CANDIDATE_SOURCE_COLS = [
    "candidate_personal",
    "candidate_repeat_all",
    "candidate_cobuy",
    "candidate_recent_global",
    "candidate_recent_location",
    "candidate_location",
    "candidate_global",
]

FEATURE_COLS = [
    "ui_tx_count",
    "ui_qty_sum",
    "ui_spend_sum",
    "ui_discount_sum",
    "ui_net_spend_sum",
    "ui_discount_rate",
    "ui_avg_price",
    "ui_avg_discount",
    "ui_avg_qty",
    "ui_last_days",
    "ui_first_days",
    "ui_active_span_days",
    "ui_tx_7d",
    "ui_tx_30d",
    "ui_tx_90d",
    "ui_qty_90d",
    "ui_share_of_user_tx",
    "ui_share_of_item_tx",
    "ui_share_of_user_qty",
    "ui_share_of_item_qty",
    "u_tx_count",
    "u_bill_nunique",
    "u_item_nunique",
    "u_qty_sum",
    "u_spend_sum",
    "u_discount_sum",
    "u_net_spend_sum",
    "u_discount_rate",
    "u_avg_price",
    "u_avg_discount",
    "u_tx_30d",
    "u_tx_90d",
    "i_tx_count",
    "i_user_nunique",
    "i_tx_30d",
    "i_tx_90d",
    "i_user_90d",
    "u_cat_l1_tx_count",
    "u_cat_l2_tx_count",
    "u_cat_l3_tx_count",
    "u_category_tx_count",
    "u_brand_tx_count",
    "u_cat_l1_tx_90d",
    "u_brand_tx_90d",
    "u_cat_l1_share",
    "u_brand_share",
    "user_item_price_ratio",
    "loc_item_tx_count",
    "loc_item_user_nunique",
    "loc_item_tx_30d",
    "loc_item_tx_90d",
    "loc_item_recent_share",
    "loc_item_share_of_item_tx",
    "candidate_personal",
    "candidate_repeat_all",
    "candidate_cobuy",
    "candidate_global",
    "candidate_recent_global",
    "candidate_location",
    "candidate_recent_location",
    "candidate_source_count",
]
