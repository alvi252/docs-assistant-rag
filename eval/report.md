# Evaluation report

- Run: 2026-10-25T13:48:17+00:00 · mode: **full**
- Models: `gpt-4.1-mini` (answers) · `text-embedding-3-small` (embeddings)
- Retrieval: top-k 5, min similarity 0.30, hybrid (vector + BM25, RRF)

## Summary

| Metric | Result |
|---|---|
| Retrieval hit-rate@5 (answerable questions) | 11/11 (100%) |
| Mean reciprocal rank of the expected document | 0.95 |
| Refusal correctness (answer vs. “I don't know”) | 14/14 (100%) |
| · answerable questions answered | 11/11 (100%) |
| · unanswerable questions refused | 3/3 (100%) |
| Expected facts present in answers | 14/15 (93%) |
| Answers citing an expected document | 11/11 (100%) |
| Latency per question (avg / p50 / max) | 1.04 s / 1.06 s / 1.37 s |
| Tokens (input / output) | 11,608 / 864 |
| Estimated cost (total / per question) | $0.0060 / $0.00043 |

## Per question

| # | Question | Expected | Top retrieved | Hit | Outcome | Correct | Facts | Latency |
|---|---|---|---|---|---|---|---|---|
| q01 | How long does standard delivery to Germany take, and how much does it cost? | `shipping-and-delivery.md` | `shipping-and-delivery.md` | yes | answered | ✅ | 2/2 | 1.22 s |
| q02 | Can I return an unopened bag of cat food, and how long do I have? | `returns-and-refunds.md` | `returns-and-refunds.md` | yes | answered | ✅ | 1/1 | 1.12 s |
| q03 | My dog refuses the PawPantry Kitchen food I opened last week. Can I get my money back? | `returns-and-refunds.md` | `returns-and-refunds.md` | yes | answered | ✅ | 2/2 | 1.17 s |
| q04 | If I pay by bank transfer, how long do you hold my order and when does it ship? | `payment-methods.md` | `payment-methods.md` | yes | answered | ✅ | 2/2 | 1.00 s |
| q05 | How do I skip or pause my Autoship, and what is the deadline for changes? | `subscription-orders.md` | `subscription-orders.md` | yes | answered | ✅ | 2/2 | 1.37 s |
| q06 | How often should I replace the filter in the AquaFlow fountain? I have two cats. | `care-guide-fountain-and-feeder.md` | `care-guide-fountain-and-feeder.md` | yes | answered | ✅ | 1/1 | 1.22 s |
| q07 | How much dry food should my 10 kg adult dog eat per day? | `feeding-guide-dry-food.md` | `feeding-guide-dry-food.md` | yes | answered | ✅ | 1/1 | 0.85 s |
| q08 | My SmartBowl feeder stopped working 18 months after delivery. Is it still covered? | `product-warranty.md` | `care-guide-fountain-and-feeder.md` | yes | answered | ✅ | 0/1 | 1.37 s |
| q09 | Can I chat with you on Saturdays, and until what time? | `contact-and-opening-hours.md` | `contact-and-opening-hours.md` | yes | answered | ✅ | 1/1 | 0.95 s |
| q10 | Combien de temps ai-je pour retourner un article ? | `returns-and-refunds.md` | `returns-and-refunds.md` | yes | answered | ✅ | 1/1 | 1.00 s |
| q11 | What is the heavy-order surcharge for a 25 kg order shipped to Spain? | `shipping-and-delivery.md` | `shipping-and-delivery.md` | yes | answered | ✅ | 1/1 | 1.18 s |
| q12 | Do you offer pet insurance? | — (not in docs) | `returns-and-refunds.md` | — | refused (model_declined) | ✅ | — | 0.71 s |
| q13 | Which vaccinations does a kitten need, and at what age? | — (not in docs) | `feeding-guide-dry-food.md` | — | refused (model_declined) | ✅ | — | 0.71 s |
| q14 | Who founded PawPantry, and in which year? | — (not in docs) | `product-warranty.md` | — | refused (model_declined) | ✅ | — | 0.71 s |

## Answers

**q01. How long does standard delivery to Germany take, and how much does it cost?**

> Standard delivery to Germany takes 3–5 business days and costs €6.90 [1].

**q02. Can I return an unopened bag of cat food, and how long do I have?**

> You can return most items, including an unopened bag of cat food, within 30 days of the delivery date as long as they are unused, unopened, and in their original packaging [3].

**q03. My dog refuses the PawPantry Kitchen food I opened last week. Can I get my money back?**

> Yes, you can get a refund if your dog refuses the PawPantry Kitchen food you opened last week, as long as you contact PawPantry within 14 days of delivery, have at least half of the bag left, and have not used the Happy Tummy Promise for the same product before. You do not need to send the food back; just send a photo of the bag showing the batch number and remaining contents to get a full refund. This applies only to the PawPantry Kitchen range [1].

**q04. If I pay by bank transfer, how long do you hold my order and when does it ship?**

> If you pay by bank transfer, we reserve your items for 7 calendar days. If the payment has not arrived by then, the order is cancelled automatically. Your order ships once the money reaches our account, which usually takes 1–2 business days after you send the transfer. Orders paid by bank transfer are dispatched once the payment has reached our account [1][3].

**q05. How do I skip or pause my Autoship, and what is the deadline for changes?**

> You can skip the next delivery with one click under My account → Autoship; the following delivery remains on schedule. To pause, you can pause the whole Autoship for up to 6 months, after which it restarts automatically with a reminder 3 days before. Changes and cancellations must be made by 23:59 CET on the day before the processing date. Orders already being processed cannot be changed but can be returned under normal return rules [1].

**q06. How often should I replace the filter in the AquaFlow fountain? I have two cats.**

> You should replace the carbon filter every 2 weeks if you have more than one pet, such as two cats, or if pets shed a lot [1].

**q07. How much dry food should my 10 kg adult dog eat per day?**

> A 10 kg adult dog should eat 160 grams of dry food per day, split into two meals. Adjust this amount by adding 10–20% for very active or working dogs, or reducing by about 10% for neutered dogs and dogs over 8 years old [1].

**q08. My SmartBowl feeder stopped working 18 months after delivery. Is it still covered?**

> The SmartBowl feeder is covered under warranty for the original warranty period or for 6 months after a repair or replacement, whichever is longer. Since your feeder stopped working 18 months after delivery, coverage depends on the length of the original warranty period. If the original warranty period is longer than 18 months, it may still be covered; otherwise, it is likely not covered. The exact original warranty duration is not specified in the provided documents [5].

**q09. Can I chat with you on Saturdays, and until what time?**

> Yes, you can chat with us on Saturdays until 17:00 CET [1][2].

**q10. Combien de temps ai-je pour retourner un article ?**

> Vous pouvez retourner la plupart des articles dans les 30 jours suivant la date de livraison, à condition qu'ils soient inutilisés, non ouverts et dans leur emballage d'origine. Pour les commandes livrées en plusieurs colis, le délai de 30 jours commence à partir de la livraison de chaque colis [1].

**q11. What is the heavy-order surcharge for a 25 kg order shipped to Spain?**

> The heavy-order surcharge for a 25 kg order shipped to Spain is €10, as orders weighing more than 20 kg are delivered by a pallet carrier and a surcharge of €10 applies for countries other than France [1].

**q12. Do you offer pet insurance?**

> The provided documents do not mention pet insurance. Please contact customer care for more information.

**q13. Which vaccinations does a kitten need, and at what age?**

> The provided documents do not contain information about kitten vaccinations or the ages at which they should be given. Please contact customer care for vaccination guidance.

**q14. Who founded PawPantry, and in which year?**

> The provided documents do not contain information about who founded PawPantry or the year it was founded. Please contact customer care for this information.
