# Business Rules — Carrefour Morocco

## Field Constraints
### orders
- order_id: required, UUID, never null
- store_id: required, must exist in store master
- status: required, values = [completed, pending, cancelled]
- total_amount: required, > 0 and < 50000 MAD
- currency: required, always MAD
- payment_method: required, values = [cash, card, mobile]

### order_line
- quantity: required, integer, > 0 and <= 500
- unit_price: required, > 0
- discount: optional, 0 to 0.40

### inventory_event
- quantity_before and quantity_after: required, >= 0
- event_type: values = [sale, restock, adjustment, return]

## Store Type Rules
- Hypermarche: 500-2000 orders/day, avg basket 250-800 MAD
- Supermarche: 200-800 orders/day, avg basket 150-400 MAD
- Proximite: 50-200 orders/day, avg basket 50-200 MAD, max item 300 MAD

## Operating Hours
- All stores: 08:00-22:00 Morocco time (UTC+1)
- Orders outside hours are suspicious

## Payment
- All transactions in MAD
- Minimum order: 5 MAD, Maximum single item: 5000 MAD
