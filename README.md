Hôm nay chúng mình sẽ giới thiệu với mọi người cách nhóm mình tư duy , xử lý vấn đề , giải quyết bài toán như thế nào nhé.
Dự án chúng mình đăng kí là xây dựng mô hình dự đoán nguy cơ mắc bệnh mãn tính.


Lý do chúng mình chọn đề tài vì :
1. CKD là bệnh nguy hiểm & âm thầm
CKD tiến triển chậm, khó nhận biết ở giai đoạn đầu và dễ dẫn đến suy thận nếu phát hiện muộn.
2. Chẩn đoán truyền thống còn hạn chế
Dựa trên nhiều chỉ số rời rạc, tính chủ quan cao và khó sàng lọc hiệu quả trên số lượng lớn bệnh nhân.
3. Cần dự đoán sớm dựa trên dữ liệu
Dữ liệu lâm sàng cho phép phát hiện sớm nguy cơ CKD, hỗ trợ theo dõi và quản lý bệnh nhân tốt hơn.
4. Machine Learning là giải pháp phù hợp
ML giúp phân tích nhiều biến cùng lúc và đưa ra dự đoán nhanh, chính xác, khách quan hơn.
5. Ý nghĩa khoa học & thực tiễn
Nghiên cứu thúc đẩy ứng dụng AI trong y tế và hỗ trợ bác sĩ cải thiện hiệu quả chẩn đoán CKD.


Thận mãn tính là CKD (Chronic Kidney Disease) = Suy giảm chức năng thận kéo dài ≥ 3 tháng, biểu hiện bởi:
Giảm mức lọc cầu thận (GFR < 60 ml/min/1.73m²)
Hoặc có dấu hiệu tổn thương thận: protein niệu, tiểu máu, bất thường hình ảnh học
CKD là bệnh mạn tính, tiến triển rất âm thầm và gần như không thể hồi phục khi ở giai đoạn muộn của bệnh
CKD gây nhiều biến chứng nguy hiểm:
Biến chứng tim mạch như suy tim
Rối loạn huyết học
Rối loạn điện giải & chuyển hóa
Suy thận giai đoạn cuối (ESRD)
Gây nguy hiểm trực tiếp đến tính mạng

Tại sao cần dự đoán CKD sớm?
CKD tiến triển âm thầm ở những giai đoạn đầu
Điều trị sớm làm chậm tiến triển và có thể bảo tồn thận
Ngăn chặn kịp thời biến chứng tim mạch, thiếu máu , ESRD
Giảm chi phí điều trị (lọc máu, ghép thận) , và giảm thiểu tài nguyên cơ sở y tế

Mục tiêu nghiên cứu :
Xây dựng mô hình dự đoán sớm và xác nhân nguy cơ mắc bệnh thận mạn tính (CKD) dựa trên dữ liệu bệnh nhân. Với độ nhạy và chính xác cao.
Ứng dụng 4 thuật toán Machine Learning phổ biến:
Logistic Regression
Decision Tree
Random Forest
K-Nearest Neighbors (KNN).

So sánh hiệu suất giữa các mô hình để xác định thuật toán phù hợp nhất cho bài toán dự đoán CKD.

Đánh giá mô hình theo các chỉ số:
Accuracy
Confution matrix
Roc&AUC
K-fold cv

Thu được mô hình tối ưu có độ nhạy ,độ chinh xác cao và ổn định để hỗ trợ sàng lọc sớm trong y tế.

Nguồn dataset chúng mình lấy trên kaggle với link: https://www.kaggle.com/datasets/mansoordaku/ckdisease 

Tên: Chronic Kidney Disease Dataset
Số mẫu: 400 bệnh nhân
Nguồn gốc: Được thu thập trong 2 tháng ở bệnh viện Fortis Gurgaon tại Ấn Độ
Số biến: 25 biến (bao gồm numeric + categorical)
Mục tiêu: Dự đoán bệnh CKD (Mắc) hoặc không CKD (Không mắc) thận mãn tính
Dạng dữ liệu: chứa cả số (numeric) và phân loại (categorical) và không có mix data
Các cột category và numeric được phân bố khá đồng đều

age       Tuổi
 bp        Huyết áp
 sg        Tỷ trọng nước tiểu
 al         Hàm lượng albumin trong nước tiểu
 su        Mức đường trong nước tiểu
 rbc      Tình trạng hồng cầu
 pc       Tình trạng bạch cầu mủ
 pcc     Cụm bạch cầu mủ
 ba       Vi khuẩn trong nước tiểu
 bgr      Đường huyết ngẫu nhiên
 bu       Ure máu
pe        Phù chân

sc             Creatinine huyết thanh
 sod          Natri
 pot           Kali
 hemo       Huyết sắc tố
 pcv           Thể tích hồng cầu đóng gói
 wc            Số lượng bạch cầu
 rc              Số lượng hồng cầu
 htn           Tăng huyết áp
 dm           Tiểu đường
 cad          Bệnh động mạch vành
 appet      Tình trạng ăn uống
 ane          Thiếu máu

 Biến mục tiêu :   classification Phân loại

Trước hết ta xem tổng quan về thống kê của các bến numeric và category
![alt text](./image/image-1.png)
![alt text](./image/image-2.png)
Ta nhận thấy rằng có 1 và cột đang bị nhầm Dtype nên ta phải biến đổi chúng về đúng Dtype
Tiếp theo chúng ta đi vào bước EDA khám phá dữ liệu để xem sự tương quan giữa các đặc trưng với nhau và với biến mục tiêu
![alt text](./image/image33.png)
Sau khi EDA thì ta nhận thấy rằng việc là các đặc trưng có độ tương quan với nhau khá rõ , Dữ liệu phân biệt nhau khá rõ ràng. 

Nhắc lại mục tiêu: Sau rất nhiều lần thử chúng tôi nhận thấy rằng đây là môt dataset đặc biệt và căn bệnh này cũng rất đặc biết 
Thứ nhất căn bệnh suy thận là một bệnh mà cực kì khó phát hiện khi còn ở giai đoạn sớm tức từ giao đoạn 2 trở về 

Thứ 2 Những triệu chứng và chỉ số sinh hóa của người mà mắc bệnh từ giai đoạn 2b trở lên thì cực kì khác biệt với những người khỏe mạnh . Vì thận là một trong những cơ quan quan trọng bặc nhất trong cơ thể nên nếu nó gập vấn đề thì chỉ số giữa người bệnh và người không mắc cực kì khác biệt
Nhưng có vài chỉ số chỉ khác biệt với người không mắc khi đã ở những giai đoạn muộn và khi ở giai đoạn muộn thì không thể phục hồi 
Thông tin được lấy từ các trang báo và có thể thấy việc đó thông qua EDA mà chúng tôi đã làm

Vậy câu hỏi đặt ra cho chúng tôi lúc này là : Nếu chỉ số sinh hóa và triệu chứng của người bị bệnh và người không bị khác nhau và tách biệt như vậy thì cần gì mô hình dự đoán nữa. Bởi vì chỉ cần nhìn vào vài chỉ số là có thể phân biệt được mà . Và đây cũng là câu hỏi khiến chúng tôi băn khoăn
Trước hết chúng ta cần phải hiểu quy trình đi khám đặc biệt là xét nghiệm với những cơ quan quan trọng trong cơ thể 
Cái này có thể tự tìm hiểu
Với số lượng bệnh nhân ngày càng trẻ hóa và lớn như hiện nay thì các bác sĩ phải làm việc với khối lượng công việc lớn dẫn đến có thể có sai sót

Vậy nên để giải quyết cho vấn đề đó chúng tôi có ý tưởng như sau : Chúng tôi sẽ làm 2 mô hình
 
Model đầu tiên sẽ ứng dụng trong việc rà soát và dự đoán sớm bệnh : lý do để làm việc này đã được đề cập phía trên

Model thứ 2 sẽ dùng để xác nhận rằng người đó bị bệnh : 
ý do để làm việc này đã được đề cập phía trên

Ý tưởng của model dự đoán sớm:
Như đã đề cập và thông qua bước EDA chúng ta nhận thầy rằng có vài chỉ số sinh hóa có thể phân biệt bệnh cực kì rõ . Và hầu như đều ở giai đoạn muộn .
Vậy nên chúng tôi sẽ lược bỏ bớt những feature mang tính muộn màng đó đi chỉ giữ lại các feature có khả năng phát hiện bệnh sớm
Để chọn được những feature đó chúng tôi dựa theo nhiều tiêu chí : 
Thường được xét nghiệm trong các cuộc khám sức khỏe định kỳ 
Không có cấu trúc mạnh mẽ  . Ví dụ như grf < 80 => ckd 
Và thông qua 1 vài trang báo và sự tham khảo . 
Sau đó chúng tôi sẽ làm những bước như train và đánh giá hiệu năng nhưng bình thường

Ý tưởng của model trong việc xác nhận :
Đối với model thứ 2 này chúng tôi sẽ train với toàn bộ thuộc tính 
Ý nghĩa của mô hình này là : Trợ giúp cho các bác sĩ có thể xác nhận và không tốn thời gian ở những bước rườm rà khác . Cho các bác sĩ có thời gian nghỉ ngơi tỉnh táo để hoàn thành tốt công việc của mình
Sau đó chúng tôi sẽ làm những bước như train và đánh giá hiệu năng nhưng bình thường

Để tránh việc tập test bị rò rỉ khi fill miss mình sẽ chia train test trước
Sau đó sẽ fill miss bằng trung vị và mode
Tiếp theo sẽ má hóa các cột category
Và cuối cùng là chuẩn hóa cho KNN và logistic

Bọn mình chọn 4 mô mình là KNN , Logistic , Decition tree và Randomforest
Vì 4 mô hình là đủ cho chúng mình có thể học tập thử nghiệm và so sánh vì 4 mô hình là không quá ít cũng không nhiều .
Bọn mình chọn những mô hình trên vì chúng thể hiện lối tư duy khác nhau trong đời sống nói chung và lĩnh vức ML nói riêng.
Đối với KNN thì là một mô hình đại diện cho lĩnh vức khoảng cách và lối tư duy khoảng cách cùng với K-means và DBscan.
Với DT và RF thì thể hiện lối tư duy logic dễ hiểu và dễ giải thích cực kì hợp lý cho việc muốn giải thích với người không học AI
Về logistic là thuộc nhóm mô hình đơn tư duy tuyến tính cùng với Linear .
Thứ 2 là 4 mô hình này dễ học , dễ hiểu và dễ giải thích với người mới học Ml 

Tiếp theo chúng mình sẽ train với 4 mô hình đó với các hyperprameter mặc định của thư viện skitlearn
Rồi sử dụng K-fold cv để đánh giá và đưa ra mô hình tốt nhất cùng với các tham số tốt nhất

Chúng mình dùng confution matrix, K-fold  cv để chọn mô hình tối ưu nhất và Shuffeld nhãn, learning curve để kiểm tra xem có vấn đề gì không 

Kết quả cho thấy mô hình logistic là sự lựa chọn phù hợp nhất với dữ liệu của chúng mình . Điều đó dễ hiểu vì khi EDA dữ liệu của chúng mình rất ít nhiễu và cực kì đẹp 2 lớp phân tách nhau gần như rõ ràng và có rất nhiều feature cực mạnh có khả năng phân biệt cực rõ hơn nữa căn bệnh này thì chỉ số sinh hóa giữa người bị bệnh và người không bị bệnh cực kì khác biệt với nhau do đó có thể hiểu tại sao logistic lại cho kết quả tối ưu nhất


kết quả của mô hình logistic cho thấy rằng Pecition 0.99 recall 1 , f1-score 0.99 , AUC 0.999
![alt text](./image/image22.png)
Để sử dụng mà không gập lỗi thì các bạn cố gắng tải Anaconda Navigator để sử dụng hoặc có thể tải hết thư viện trong file requirement.txt 

Cấu trức thư mục bao gồm 
App : Nhưng chỉ có file readme.md vì mọi thứ thư load dữ liệu cho đến EDA , train và đánh giá mô hình đã có trong demo rồi
Demo : Bao gồm những file py và ipynb gồm file chính kds.ipynb và các file ipynb khác để hiểu rõ feature và file demo.py là file chạy demo dự đoán
Model : là các file pkl dùng để lưu lại model đã chọn và tiền xử lý khi demo
Report : là báo cáo và pp về dự án
Templates : là phần giao diện demo
.gitignore là những phân không được tải lên github

