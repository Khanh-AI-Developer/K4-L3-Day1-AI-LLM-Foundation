# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)

Khi temperature tăng từ 0.0 lên 1.5, câu trả lời sẽ đa dạng, sáng tạo hơn nhưng khó dự đoán. Ở temperature 0.0, phản hồi ổn định, ngắn gọn và tập trung vào sự thậ trong khi ở mức 1.0–1.5, cách diễn đạt phong phú hơn nhưng nhiều trường hợp xuất hiện chi tiết lan man, kém chính xác.


### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**

Tôi sẽ đặt temperature khoảng 0.1–0.3 cho chatbot hỗ trợ khách hàng. Vì chọn mức temperature như vậy sẽ đảm bảo chatbot luôn đưa ra câu trả lời chuẩn xác, ngăn chặn việc bot tự bịa ra chính sách, giá cả hoặc thông tin không có thật, buộc chatbot phải dựa chặt chẽ vào cơ sở dữ liệu nội bộ thay vì tự sáng tạo.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**

Tổng số token đầu ra mỗi ngày là: 10.000 × 3 × 350 = 10.500.000 token.

Nếu sử dụng mức giá trong bài thực hành là 10 USD/1 triệu output token cho GPT-4o và 0,60 USD/1 triệu output token cho GPT-4o-mini, chi phí đầu ra ước tính:

GPT-4o: 10,5 × 10 = 105 USD/ngày.
GPT-4o-mini: 10,5 × 0,60 = 6,30 USD/ngày.

Như vậy, GPT-4o đắt hơn khoảng 16,7 lần đối với phần token đầu ra. GPT-4o xứng đáng với chi phí khi xử lý yêu cầu phức tạp như phân tích hợp đồng, lập luận nhiều bước hoặc hỗ trợ kỹ thuật chuyên sâu. GPT-4o-mini phù hợp với các tác vụ số lượng lớn và đơn giản như phân loại câu hỏi, tóm tắt tài liệu ngắn. 

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)

Phản hồi của giáo viên sẽ ngắn gọn, sử dụng từ ngữ đơn giản và ví dụ trực quan, trong khi chuyên gia sẽ đưa ra câu trả lời dài hơn với các thuật ngữ phức tạp như chính sách tài chính, hợp đồng thông minh. System prompt ảnh hưởng đến hành vi của mô hình bằng cách thiết lập bộ quy tắc nền tảng về nhân dạng (persona), giọng điệu và đối tượng mục tiêu ngay từ đầu. Thông qua đó, nó định hướng cơ chế chú ý (attention mechanism) và điều chỉnh xác suất sinh từ vựng, buộc mô hình phải lọc bỏ các thông tin lạc đề để đóng vai một cách hoàn hảo.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**

Phương pháp số từ / 0.75 ước tính khoảng 133 token, trong khi count_tokens bằng tiktoken cho kết quả khoảng 180 token. Mức chênh lệch là:

(180 − 133) / 133 × 100 ≈ 35,3%.

Tiếng Việt thường tốn nhiều token hơn vì một từ có dấu hoặc một từ ghép có thể bị tokenizer chia thành nhiều token nhỏ như dấu câu, ngữ âm, vần. Các tokenizer cũng thường được tối ưu tốt hơn cho những mẫu từ xuất hiện phổ biến trong dữ liệu tiếng Anh.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)

Streaming quan trọng nhất khi model tạo câu trả lời dài như viết nội dung, giải thích bài học hoặc sinh mã nguồn, vì người dùng có thể nhìn thấy kết quả ngay mà không phải chờ toàn bộ phản hồi hoàn thành. Non-streaming phù hợp khi phản hồi ngắn, khi ứng dụng cần nhận đủ kết quả để kiểm tra hoặc chuyển đổi sang JSON, hoặc khi nội dung phải được kiểm duyệt hoàn chỉnh trước khi hiển thị.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**

Exponential backoff tăng dần thời gian chờ sau mỗi lần thất bại, ví dụ 1, 2, 4 rồi 8 giây, nhờ đó giảm số lượng request gửi đến API khi hệ thống đang quá tải và tạo thời gian cho dịch vụ phục hồi. Nếu hàng nghìn client đều retry sau một khoảng cố định như 1 giây, chúng có thể đồng loạt gửi lại request khiến API tiếp tục quá tải.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**

System prompt:

“Bạn là trợ lý học tập AI thân thiện dành cho sinh viên Việt Nam. Hãy trả lời bằng tiếng Việt, giải thích rõ ràng từ cơ bản đến nâng cao và sử dụng ví dụ thực tế khi phù hợp. Ưu tiên câu trả lời ngắn gọn, có cấu trúc và cấm bịa thông tin. Nếu không chắc chắn, hãy nói rõ giới hạn và đề nghị người dùng kiểm tra lại nguồn đáng tin cậy.”

Tôi yêu cầu “trả lời bằng tiếng Việt” để trợ lý nhất quán với nhóm người dùng mục tiêu. Cụm từ “cấm bịa thông tin” và “nói rõ khi không chắc chắn” giúp giảm nguy cơ model trình bày thông tin sai với giọng điệu quá tự tin. Yêu cầu “từ cơ bản đến nâng cao” giúp câu trả lời phù hợp với sinh viên đang học một kiến thức mới.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**

Hạn chế lớn nhất của trợ lý hiện tại là chỉ giữ một số lượt hội thoại gần nhất nên có thể quên mục tiêu, trình độ và nội dung người dùng đã trao đổi trước đó. Tôi đề xuất bổ sung bộ nhớ dài hạn bằng cách tóm tắt các thông tin quan trọng của cuộc hội thoại và lưu chúng theo user_id trong PostgreSQL hoặc một vector database. Khi có yêu cầu mới, hệ thống truy xuất những thông tin liên quan rồi đưa chúng vào context trước khi gọi LLM. Cần giới hạn loại dữ liệu được lưu, mã hóa thông tin nhạy cảm và cho phép người dùng xóa bộ nhớ.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
