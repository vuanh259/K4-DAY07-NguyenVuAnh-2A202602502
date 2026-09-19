# Báo Cáo Nhóm — Lab 7: Embedding & Vector Store

**Nhóm:** Magician
**Thành viên:** Nguyễn Vũ Anh, Phạm Quang Đạt, Nguyễn Thanh Duy, Trương Việt Anh
**Ngày:** 19/09/2026

> Báo cáo tổng hợp này dùng cùng corpus và cùng 5 truy vấn benchmark mà các thành viên đã chạy trong báo cáo cá nhân.

## 1. Lựa chọn tài liệu (Document Set Quality) — Nhóm (10 điểm)

### Chủ đề và lý do chọn

**Chủ đề:** Quy định và dịch vụ đào tạo đại học: đăng ký/rút học phần, phúc khảo và cảnh báo học vụ.

Đây là miền dữ liệu phù hợp cho RAG vì câu trả lời cần chính xác về mốc thời gian, tín chỉ và đối tượng áp dụng. Văn bản có cấu trúc heading/điều khoản rõ ràng, đồng thời có các tài liệu cùng chủ đề nhưng khác `audience`, giúp kiểm chứng metadata filtering.

### Danh sách tài liệu

| # | Tài liệu | Nguồn công khai | Phiên bản / ngày lấy | Ký tự | Metadata chính |
|---|---|---|---|---:|---|
| 1 | `tvu-course-registration.md` | [Sổ tay SV TVU](https://cmp.tvu.edu.vn/so-tay-sinh-vien/) | Sổ tay SV 2026 / 19-09-2026 | 2.017 | `audience=student`, `category=registration`, `university=TVU` |
| 2 | `tvu-academic-warning.md` | [Sổ tay SV TVU](https://cmp.tvu.edu.vn/so-tay-sinh-vien/) | Sổ tay SV 2026 / 19-09-2026 | 1.691 | `audience=student`, `category=warning`, `university=TVU` |
| 3 | `tvu-grade-appeal.md` | [Sổ tay SV TVU](https://cmp.tvu.edu.vn/so-tay-sinh-vien/) | Sổ tay SV 2026 / 19-09-2026 | 1.479 | `audience=student`, `category=grade-appeal`, `university=TVU` |
| 4 | `qtu-academic-affairs-overview.md` | [QTU — QĐ 95](https://qtu.edu.vn/qd-95-ban-hanh-quy-dinh-ve-cong-tac-hoc-vu-tai-truong-dai-hoc-quang-trung/) | 95/QĐ-ĐHQT / 19-09-2026 | 1.505 | `audience=all`, `category=overview`, `university=QTU` |
| 5 | `tdmu-grade-appeal.md` | [TDMU — QT.09](https://tdmu.edu.vn/hinh/thuvien/taptin/2-6-2025-4-42-24-pm06-BKTKDDBCL-QT.09-Phuc%20khao%20Bai%20KTr.pdf) | QT.09 Lần 01 / 19-09-2026 | 1.325 | `audience=student`, `category=grade-appeal`, `university=TDMU` |
| 6 | `tdmu-grade-appeal-operations.md` | [TDMU — QT.09](https://tdmu.edu.vn/hinh/thuvien/taptin/2-6-2025-4-42-24-pm06-BKTKDDBCL-QT.09-Phuc%20khao%20Bai%20KTr.pdf) | QT.09 Lần 01 / 19-09-2026 | 1.392 | `audience=staff`, `category=grade-appeal-operations`, `university=TDMU` |
| 7 | `tnut-advanced-registration.md` | [TNUT — QĐ 3571](https://fit.tnut.edu.vn/bai-viet/quy-che-dao-tao-trinh-do-dai-hoc-cho-chuong-trinh-tien-tien-nam-2022-176) | 3571/QĐ-ĐHKTCN / 19-09-2026 | 1.363 | `audience=student`, `category=registration-advanced`, `university=TNUT` |
| 8 | `tnut-advanced-withdrawal-assessment.md` | [TNUT — QĐ 3571](https://fit.tnut.edu.vn/bai-viet/quy-che-dao-tao-trinh-do-dai-hoc-cho-chuong-trinh-tien-tien-nam-2022-176) | 3571/QĐ-ĐHKTCN / 19-09-2026 | 1.244 | `audience=student`, `category=withdrawal-advanced`, `university=TNUT` |
| 9 | `ufm-course-registration.md` | [UFM — QĐ 1329](https://pdt.ufm.edu.vn/dulieu/quiche/1329_Quy_che_dao_tao_tin_chi_tu_khoa_2021.htm) | 1329/QĐ-ĐHTCM / 19-09-2026 | 1.329 | `audience=student`, `category=registration`, `university=UFM` |
| 10 | `ufm-assessment.md` | [UFM — QĐ 1329](https://pdt.ufm.edu.vn/dulieu/quiche/1329_Quy_che_dao_tao_tin_chi_tu_khoa_2021.htm) | 1329/QĐ-ĐHTCM / 19-09-2026 | 1.225 | `audience=student`, `category=assessment`, `university=UFM` |

Mỗi tài liệu được lưu kèm `source_url`, `retrieved_at` và `document_version`; các nguồn đều công khai, không chứa dữ liệu cá nhân hay thông tin nội bộ.

### Metadata schema

| Trường | Ví dụ | Mục đích truy xuất |
|---|---|---|
| `audience` | `student`, `staff`, `all` | Lọc đúng đối tượng áp dụng trước khi xếp hạng. |
| `category` | `registration`, `grade-appeal` | Giảm nhiễu giữa các nghiệp vụ. |
| `university` | `TVU`, `TDMU`, `TNUT`, `UFM` | Phân biệt các quy định có từ khóa tương tự. |
| `document_version` | `QT.09 Lần 01` | Kiểm tra hiệu lực văn bản. |
| `source_url`, `retrieved_at` | URL, `2026-09-19` | Truy vết nguồn và độ mới dữ liệu. |

## 2. Thiết kế chiến lược (Strategy Design) — Nhóm (15 điểm)

### Baseline

Trên cùng corpus, ba chiến lược cơ sở cho thấy đánh đổi rõ ràng: Fixed-size có chunk đồng đều nhưng có thể cắt ngang câu; Sentence giữ nguyên câu nhưng đôi khi tách tiêu đề khỏi nội dung; Recursive/heading-aware theo cấu trúc Markdown giữ được ngữ cảnh điều khoản tốt hơn.

### Chiến lược của từng thành viên

| Thành viên | Chiến lược | Cấu hình / quy mô | Lý do và nhận xét |
|---|---|---|---|
| Nguyễn Vũ Anh | `HeadingRecursiveChunker` | Chia theo heading, sau đó `RecursiveChunker` khi section quá dài | Gắn lại heading cho mọi mảnh con, nên câu điều khoản vẫn biết đang nói về quy định nào. Phù hợp với văn bản pháp quy có cấu trúc phân cấp. |
| Phạm Quang Đạt | `FixedSizeChunker` | `chunk_size=800`, `overlap=150`; 19 chunks, TB 616,47 ký tự | Nhanh, đơn giản và overlap giảm mất thông tin tại điểm cắt; tuy nhiên không khắc phục hoàn toàn mất ngữ cảnh nguồn. |
| Nguyễn Thanh Duy | `SentenceChunker` | `max_sentences_per_chunk=3`; 32 chunks, TB 322,28 ký tự | Bảo toàn câu và dấu câu, hữu ích khi một điều khoản gồm 1–3 câu; có thể tách chi tiết liên quan sang chunk kế tiếp. |
| Trương Việt Anh | `HeadingSectionChunker` | `chunk_size=500`; 34 chunks, TB 352,21 ký tự | Tách section theo heading và đệ quy section dài; heading được giữ trong chunk để tăng tín hiệu tên trường/chủ đề. |

### So sánh kết quả cá nhân

| Chiến lược | Kết quả benchmark | Điểm mạnh | Hạn chế |
|---|---:|---|---|
| Fixed-size (Phạm Quang Đạt) | 7/10; 4/5 có chunk liên quan trong top-3 | Cấu hình 800/150 cải thiện so với cửa sổ nhỏ. | Q1 không có tài liệu đúng trong top-3; Q3 đúng tài liệu nhưng thiếu chi tiết. |
| Sentence (Nguyễn Thanh Duy) | 5/5 top-1 liên quan; Q2 thiếu chi tiết lệ phí ở chunk kế tiếp | Không cắt gãy câu; Q1, Q3–Q5 đầy đủ bằng chứng. | Q2 chia phần lệ phí khỏi chunk chứa nơi nộp và thời hạn. |
| Heading/section (Trương Việt Anh) | 5/5 ở mức tài liệu và nội dung; 10/10 | Giữ heading và cấu trúc section, bao phủ đủ thuật ngữ đáp án trong top-3. | Cần Markdown có heading chuẩn. |
| Heading-recursive (Nguyễn Vũ Anh) | Theo chấm nội dung, phát hiện sai-section ở truy vấn cảnh báo học vụ | Làm rõ khác biệt giữa “đúng tài liệu” và “đúng chunk chứa đáp án”. | Cần đánh giá theo `answer_terms`, không chỉ theo `doc_id`. |

Kết luận: Heading-aware là lựa chọn ưu tiên cho corpus này; SentenceChunker là phương án đơn giản nhưng hiệu quả khi điều khoản ngắn. Fixed-size hữu ích làm baseline, song không nên dùng độc lập cho câu hỏi yêu cầu nhiều mốc số liệu.

## 3. Câu hỏi đánh giá & chất lượng truy xuất — Nhóm (10 điểm)

| # | Câu hỏi benchmark | Gold answer rút gọn | Nguồn/chunk kỳ vọng |
|---|---|---|---|
| 1 | TVU: thời hạn rút học phần và xử lý tự ý bỏ học từ tuần thứ ba? | Rút trong 2 tuần đầu học kỳ chính; từ tuần thứ ba tự ý bỏ học nhận điểm F. | `tvu-course-registration::c03` |
| 2 | TDMU: nộp đơn phúc khảo ở đâu, trong bao lâu và lệ phí thế nào? *(filter `audience=student`)* | Nộp đơn BM.01 về bộ môn quản lý đề cương trong 7 ngày từ khi công bố điểm; thực hiện theo quy định lệ phí phúc khảo. | `tdmu-grade-appeal::c01–c02` |
| 3 | TNUT chương trình tiên tiến: giới hạn tín chỉ học kỳ chính? | Năm có 3 kỳ chính: 8–16 tín chỉ; năm có 2 kỳ chính: 10–24 tín chỉ. | `tnut-advanced-registration::c02` |
| 4 | UFM: trước khi đăng ký cần tìm hiểu gì và nhờ ai tư vấn? | Chương trình, đề cương, điều kiện, kế hoạch và thời khóa biểu; hỏi cố vấn học tập. | `ufm-course-registration::c02` |
| 5 | TVU: ngưỡng cảnh báo ĐTBTL và số tín chỉ F? | <1,20; <1,40; <1,60; <1,80 theo năm; F tích lũy vượt 24 tín chỉ. | `tvu-academic-warning::c01` |

**Kết quả tổng hợp:** Sentence và heading/section đều đưa được bằng chứng liên quan cho 5/5 truy vấn. Fixed-size đạt 4/5. Riêng Q2 cho thấy filter `{"audience": "student"}` cần thiết: nó loại `tdmu-grade-appeal-operations.md` dành cho `staff`, tránh trộn quy trình nội bộ với hướng dẫn cho sinh viên.

**Phân tích lỗi:** Fixed-size nhầm Q1 sang tài liệu phúc khảo TVU và không lấy được tài liệu rút học phần trong top-3. Với SentenceChunker, Q2 lấy đúng chunk nộp đơn/7 ngày nhưng thông tin lệ phí nằm ở chunk kế tiếp. Biện pháp cải thiện: chunk theo heading có overlap theo câu, dùng `top_k` đủ lớn, và chấm theo các `answer_terms` bắt buộc thay vì chỉ kiểm tra `doc_id`.

## 4. Demo & bài học nhóm — Nhóm (5 điểm)

Demo trình bày cùng corpus, cùng 5 truy vấn và 3 kiểu chunking. Ba kết luận chính:

1. Cấu trúc heading/section giúp mỗi chunk bảo toàn chủ đề và nguồn gốc điều khoản.
2. Metadata pre-filtering là lớp bảo vệ trước khi retrieval, đặc biệt với hai tài liệu TDMU khác đối tượng áp dụng.
3. Đánh giá retrieval phải kiểm tra nội dung chunk chứa đáp án; đúng `doc_id` chưa đủ để agent trả lời đúng.

Nếu làm lại, nhóm sẽ bổ sung `effective_date` và `status=active`, đồng thời chuyển bảng số liệu trong quy chế sang key-value/JSON để các ngưỡng số được truy xuất ổn định hơn.

## Tự đánh giá

| Tiêu chí | Điểm tự đánh giá |
|---|---:|
| Lựa chọn tài liệu | 10/10 |
| Thiết kế chiến lược | 15/15 |
| Chất lượng truy xuất | 10/10 |
| Thuyết trình | 5/5 |
| **Tổng** | **40/40** |
