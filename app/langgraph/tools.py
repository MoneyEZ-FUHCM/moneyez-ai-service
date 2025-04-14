import re
from typing import Optional
from datetime import datetime, timedelta, timezone
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
import requests
import json
from langchain.prompts import PromptTemplate

def get_user_subcategories(user_id):
    """Fetch subcategories for a given user ID from the API."""
    if not user_id:
        print("[TOOLS] Warning: user_id is empty")
        # Fallback to a default user_id or return empty list
        return "No subcategories available"
    
    print(f"\n[TOOLS] get_user_subcategories called for user: {user_id}")
    headers = {"X-External-Secret": "thisIsSerectKeyPythonService"}
    url = f"https://easymoney.anttravel.online/api/v1/external-services?command=get_subcategories&query=user_id={user_id}"

    print(f"[TOOLS] Making API request to: {url}")
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        print(f"[TOOLS] API request successful, status code: {response.status_code}")
        subcategories = response.json().get("data", [])
        print(f"[TOOLS] Retrieved {len(subcategories)} subcategories")
        formatted_subcategories = []
        for sc in subcategories:
            formatted_subcategories.append(f"""Là một danh mục con nằm trong danh mục {sc.get("categoryName")}, danh mục này có tên là {sc.get("name")}, mã danh mục là {sc.get("code")}, mô tả là {sc.get("description")}""")
        return "\n".join(formatted_subcategories)
    else:
        print(f"[TOOLS] API request failed, status code: {response.status_code}")
        print(f"[TOOLS] API request failed, response: {response.data.message}")
        return None

def parse_response(response):
    """Parse the JSON response from the LLM."""
    try:
        response = re.sub(r"```", "", response)
        response = response.replace("\n", "").replace("json", "")
        parsed_response = json.loads(response)
        return parsed_response
    except ValueError:
        print("[ERROR] Failed to parse JSON response")
        return None

@tool(return_direct=False)
def user_input_expense(user_query: str):
    """Process user input for expenses and classify them."""
    from app.add_langgraph_route import userId
    print(f"[TOOLS] user_input_expense called with user_id: {userId}")

    if userId is None:
        return "User ID is not set. Please provide a valid user ID."
    
    # Sử dụng múi giờ Việt Nam (UTC+7)
    vietnam_tz = timezone(timedelta(hours=7))
    now = datetime.now(vietnam_tz)
    current_date = now.strftime('%Y-%m-%d')
    current_time = now.strftime('%H:%M:%S')
    current_datetime = f"{current_date}T{current_time}"
    
    subcategories = get_user_subcategories(userId)
    subllm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    template = PromptTemplate(
        input_variables=["subcategories", "user_query", "current_date", "current_time", "current_datetime"],
        template="""
        Bạn là một trợ lý tài chính thông minh. Nhiệm vụ của bạn là phân tích chi tiêu của người dùng và phân loại vào danh mục thích hợp.
        
        Dưới đây là các danh mục chi tiêu có sẵn:
        {subcategories}
        
        Thời gian hiện tại là: {current_datetime} (theo định dạng ISO: YYYY-MM-DDThh:mm:ss)
        
        Người dùng vừa nhập: "{user_query}"
        
        Hãy phân tích thông tin này và trả về kết quả dưới dạng JSON với cấu trúc sau:
        {{
            "amount": [số tiền chi tiêu, chỉ bao gồm con số],
            "subcategory_code": [mã danh mục phù hợp nhất],
            "description": [mô tả chi tiêu],
            "transaction_datetime": [thời gian giao dịch, định dạng YYYY-MM-DDThh:mm:ss]
        }}
        
        Chỉ trả về đúng định dạng JSON yêu cầu, không thêm bất kỳ giải thích nào khác.
        Phần mô tả phân tích từ tin nhắn người dùng nhập vào, ghi ngắn gọn phù hợp.
        Lưu ý:
        - Các từ ngữ có thể chỉ tiền như k, lít, củ, xị,...... của ngôn ngữ tiếng việt đơn vị là VNĐ, nhỏ nhất là 1000 VNĐ
        - Người dùng có thể nhập không có đơn vị thì mặc định hiểu là nghìn VNĐ. ví dụ: 100 là 100.000 VNĐ
        - Nếu không thể xác định được số tiền hoặc danh mục, hãy gán giá trị null cho trường tương ứng
        - Phần thời gian giao dịch:
          + Nếu người dùng ghi rõ ngày và giờ (ví dụ: "hôm qua lúc 3h chiều", "9h sáng ngày 10/4",...), hãy trích xuất và chuyển về định dạng YYYY-MM-DDThh:mm:ss
          + Nếu người dùng chỉ đề cập đến ngày (ví dụ: "hôm qua", "ngày 10/4"), hãy sử dụng ngày đó và thời gian hiện tại
          + Nếu người dùng chỉ đề cập đến giờ (ví dụ: "3h chiều", "9h sáng"), hãy sử dụng ngày hiện tại và giờ đó
          + Nếu không có thông tin về ngày và giờ, sử dụng thời gian hiện tại ({current_datetime})
        """
    )
    prompt = template.format(
        subcategories=subcategories,
        user_query=user_query,
        current_date=current_date,
        current_time=current_time,
        current_datetime=current_datetime
    )
    response = subllm.invoke(prompt)
    print(f"[TOOLS] LLM response: {response.content}")
    parsed_response = parse_response(response.content)
    
    # Xử lý thời gian giao dịch
    transaction_datetime = parsed_response.get("transaction_datetime", current_datetime)
    # Đảm bảo thời gian giao dịch có giá trị
    if not transaction_datetime:
        transaction_datetime = current_datetime
    print(f"[TOOLS] Transaction datetime: {transaction_datetime}")

    url = "https://easymoney.anttravel.online/api/v1/external-services"
    headers = {"X-External-Secret": "thisIsSerectKeyPythonService"}
    body = {
        "command": "create_transaction_v2",
        "data": {
            "UserId": userId,
            "Amount": parsed_response.get("amount"),
            "SubcategoryCode": parsed_response.get("subcategory_code"),
            "Description": parsed_response.get("description"),
            "TransactionDate": transaction_datetime  # Thời gian giao dịch đầy đủ (ngày + giờ)
        }
    }
    response = requests.post(url, headers=headers, json=body)
    print(f"[TOOLS] Request body: {body}")
    
    if response.status_code in [200, 201]:
        print(f"[TOOLS] API request successful, status code: {response.status_code}")
        
        try:
            # Parse API response data
            response_data = json.loads(response.content)
            transaction_data = response_data.get("data", {})
            
            # Extract transaction details from response
            amount = transaction_data.get("amount", parsed_response.get("amount", 0))
            subcategory_name = transaction_data.get("subcategoryName", "")
            description = transaction_data.get("description", parsed_response.get("description", ""))
            transaction_date = transaction_data.get("transactionDate", transaction_datetime)
            transaction_type = transaction_data.get("type", "EXPENSE").upper()
            
            # Format response for user
            amount_str = format_currency(amount)
            datetime_str = format_user_friendly_datetime(transaction_date)
            transaction_id = transaction_data.get("id", "")
            
            # Build user-friendly response based on transaction type
            if transaction_type == "EXPENSE":
                result = f"✅ Đã ghi nhận khoản chi tiêu {amount_str} vào {datetime_str}"
                if subcategory_name:
                    result += f" cho danh mục {subcategory_name}"
                if description:
                    result += f" với mô tả: {description}"
            else:  # INCOME
                result = f"✅ Đã ghi nhận khoản thu nhập {amount_str} vào {datetime_str}"
                if subcategory_name:
                    result += f" cho danh mục {subcategory_name}"
                if description:
                    result += f" với mô tả: {description}"
            
            # Add transaction ID for reference
            if transaction_id:
                result += f"\nMã giao dịch: {transaction_id}"
                
            return result
            
        except Exception as e:
            print(f"[TOOLS] Error parsing transaction response: {str(e)}")
            # Fallback to basic response if parsing fails
            amount_str = format_currency(parsed_response.get("amount", 0))
            return f"✅ Đã ghi nhận khoản giao dịch {amount_str} thành công"
    else:
        print(f"[TOOLS] API request failed, status code: {response.status_code}")
        print(f"[TOOLS] API request failed, response: {response.content}")
        return f"❌ Đã xảy ra lỗi khi ghi nhận giao dịch. Vui lòng thử lại sau."

@tool(return_direct=False)
def get_transaction_history(date_range: Optional[str] = None):
    """
    Lấy lịch sử giao dịch (thu, chi) của người dùng.
    Có thể truyền vào khoảng thời gian (tùy chọn) theo định dạng:
    - "hôm nay": lấy giao dịch của ngày hôm nay
    - "hôm qua": lấy giao dịch của ngày hôm qua
    - "tuần này": lấy giao dịch của 7 ngày gần nhất
    - "tháng này": lấy giao dịch của 30 ngày gần nhất
    - "dd-mm-yyyy to dd-mm-yyyy": lấy giao dịch trong khoảng thời gian cụ thể

    Nếu không chỉ định khoảng thời gian, mặc định lấy giao dịch của 7 ngày gần nhất.
    """
    from app.add_langgraph_route import userId
    print(f"[TOOLS] get_transaction_history called with user_id: {userId}")
    
    if not userId:
        return "User ID không được thiết lập. Vui lòng đăng nhập để xem lịch sử giao dịch."
    
    # Sử dụng múi giờ Việt Nam (UTC+7)
    vietnam_tz = timezone(timedelta(hours=7))
    today = datetime.now(vietnam_tz)
    
    print(f"[TOOLS] Current time in Vietnam: {today.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    
    from_date = (today - timedelta(days=7)).strftime('%Y-%m-%d')  # Mặc định 7 ngày gần nhất
    to_date = today.strftime('%Y-%m-%d')
    
    if date_range:
        date_range = date_range.lower().strip()
        if "hôm nay" in date_range:
            from_date = today.strftime('%Y-%m-%d')
        elif "hôm qua" in date_range:
            yesterday = today - timedelta(days=1)
            from_date = yesterday.strftime('%Y-%m-%d')
            to_date = yesterday.strftime('%Y-%m-%d')
        elif "tuần này" in date_range:
            from_date = (today - timedelta(days=7)).strftime('%Y-%m-%d')
        elif "tháng này" in date_range:
            from_date = (today - timedelta(days=30)).strftime('%Y-%m-%d')
        elif " to " in date_range:
            # Xử lý khoảng thời gian dạng "dd-mm-yyyy to dd-mm-yyyy"
            try:
                date_parts = date_range.split(" to ")
                date_format = '%d-%m-%Y'
                
                # Chuyển đổi định dạng ngày từ dd-mm-yyyy sang yyyy-mm-dd
                # Đảm bảo ngày giờ là ở múi giờ Việt Nam
                start_date = datetime.strptime(date_parts[0].strip(), date_format).replace(tzinfo=vietnam_tz)
                end_date = datetime.strptime(date_parts[1].strip(), date_format).replace(tzinfo=vietnam_tz)
                
                from_date = start_date.strftime('%Y-%m-%d')
                to_date = end_date.strftime('%Y-%m-%d')
            except Exception as e:
                print(f"[TOOLS] Error parsing date range: {str(e)}")
                return f"Định dạng ngày không hợp lệ. Vui lòng sử dụng định dạng dd-mm-yyyy to dd-mm-yyyy. Lỗi: {str(e)}"
    
    # Gọi API để lấy lịch sử giao dịch
    print(f"[TOOLS] Fetching transaction history for user {userId} from {from_date} to {to_date}")
    
    headers = {"X-External-Secret": "thisIsSerectKeyPythonService"}
    url = f"https://easymoney.anttravel.online/api/v1/external-services?command=get_transaction_histories_user&query=user_id={userId},from_date={from_date},to_date={to_date}"
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise exception nếu HTTP error
        
        data = response.json()
        transactions = data.get("data", [])
        
        if not transactions:
            return f"Không tìm thấy giao dịch nào trong khoảng thời gian từ {from_date} đến {to_date}."
        
        # Phân loại giao dịch theo thu/chi
        income_transactions = [t for t in transactions if t.get("type") == "INCOME"]
        expense_transactions = [t for t in transactions if t.get("type") == "EXPENSE"]
        
        total_income = sum(t.get("amount", 0) for t in income_transactions)
        total_expense = sum(t.get("amount", 0) for t in expense_transactions)
        balance = total_income - total_expense
        
        # Định dạng kết quả
        result = f"📊 **Báo cáo giao dịch từ {from_date} đến {to_date}**\n\n"
        
        # Thống kê tổng quan
        result += f"**Tổng quan:**\n"
        result += f"- Tổng thu: {format_currency(total_income)}\n"
        result += f"- Tổng chi: {format_currency(total_expense)}\n"
        result += f"- Số dư: {format_currency(balance)}\n\n"
        
        # Chi tiết giao dịch thu
        if income_transactions:
            result += f"**Các khoản thu ({len(income_transactions)}):**\n"
            for t in income_transactions:
                date = format_date(t.get("transactionDate", ""))
                result += f"- {date}: {t.get('description', 'Không có mô tả')} ({t.get('subcategoryName', 'Không có danh mục')}) - {format_currency(t.get('amount', 0))}\n"
            result += "\n"
        
        # Chi tiết giao dịch chi
        if expense_transactions:
            result += f"**Các khoản chi ({len(expense_transactions)}):**\n"
            for t in expense_transactions:
                date = format_date(t.get("transactionDate", ""))
                result += f"- {date}: {t.get('description', 'Không có mô tả')} ({t.get('subcategoryName', 'Không có danh mục')}) - {format_currency(t.get('amount', 0))}\n"
        
        return result
        
    except requests.exceptions.RequestException as e:
        print(f"[TOOLS] API request failed: {str(e)}")
        return f"Đã xảy ra lỗi khi lấy lịch sử giao dịch: {str(e)}"

@tool(return_direct=False)
def get_current_spending_model():
    """
    Lấy thông tin về mô hình chi tiêu đang được sử dụng bởi người dùng hiện tại.
    Trả về thông tin chi tiết về mô hình chi tiêu bao gồm tên, mô tả, các danh mục và tỷ lệ phân bổ.
    """
    from app.add_langgraph_route import userId
    print(f"[TOOLS] get_current_spending_model called with user_id: {userId}")
    
    if not userId:
        return "User ID không được thiết lập. Vui lòng đăng nhập để xem mô hình chi tiêu."
    
    # Gọi API để lấy thông tin mô hình chi tiêu hiện tại
    print(f"[TOOLS] Fetching current spending model for user {userId}")
    
    headers = {"X-External-Secret": "thisIsSerectKeyPythonService"}
    url = f"https://easymoney.anttravel.online/api/v1/external-services?command=get_user_spending_model&query=user_id={userId}"
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise exception nếu HTTP error
        
        data = response.json()
        
        if data.get("status") != 200 or not data.get("data"):
            print(f"[TOOLS] API returned error or empty data: {data.get('message')}")
            return "Không tìm thấy mô hình chi tiêu nào đang được sử dụng. Vui lòng thiết lập mô hình chi tiêu trước."
        
        model_data = data.get("data", {})
        
        # Trích xuất thông tin cần thiết
        name = model_data.get("name", "Không có tên")
        description = model_data.get("description", "Không có mô tả")
        start_date = format_user_friendly_date(model_data.get("startDate", "").split("T")[0] if "T" in model_data.get("startDate", "") else model_data.get("startDate", ""))
        end_date = format_user_friendly_date(model_data.get("endDate", "").split("T")[0] if "T" in model_data.get("endDate", "") else model_data.get("endDate", ""))
        categories = model_data.get("categories", [])
        
        # Loại bỏ HTML tags từ mô tả để hiển thị đẹp hơn
        clean_description = re.sub(r'<[^>]*>', '', description)
        
        # Định dạng phản hồi thân thiện với người dùng
        result = f"📊 **Mô hình chi tiêu hiện tại: {name}**\n\n"
        
        # Thông tin cơ bản
        result += f"**Thời gian áp dụng:** Từ {start_date} đến {end_date}\n\n"
        
        # Lấy mô tả trực tiếp từ API
        result += f"**Mô tả:** {clean_description}\n\n"
        
        # Các danh mục
        if categories and len(categories) > 0:
            result += "\n**Các danh mục trong mô hình:**\n"
            for cat in categories:
                result += f"- {cat.get('name')}\n"
        
        # Lời khuyên
        result += "\n**Lưu ý:** Bạn nên phân bổ các khoản chi tiêu theo tỷ lệ mô hình để quản lý tài chính hiệu quả."
        
        return result
        
    except requests.exceptions.RequestException as e:
        print(f"[TOOLS] API request failed: {str(e)}")
        return f"Đã xảy ra lỗi khi lấy thông tin mô hình chi tiêu: {str(e)}"

@tool(return_direct=False)
def get_available_spending_models():
    """
    Lấy danh sách tất cả các mô hình chi tiêu được hệ thống hỗ trợ.
    Trả về thông tin chi tiết về mỗi mô hình bao gồm ID, tên và mô tả.
    """
    print(f"[TOOLS] get_available_spending_models called")
    
    # Gọi API để lấy danh sách các mô hình chi tiêu
    headers = {"X-External-Secret": "thisIsSerectKeyPythonService"}
    url = "https://easymoney.anttravel.online/api/v1/external-services?command=get_spending_models"
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise exception nếu HTTP error
        
        data = response.json()
        
        if data.get("status") != 200 or not data.get("data"):
            print(f"[TOOLS] API returned error or empty data: {data.get('message')}")
            return "Không thể lấy được danh sách mô hình chi tiêu. Vui lòng thử lại sau."
        
        models = data.get("data", [])
        
        # Nếu không có mô hình nào
        if not models:
            return "Hiện tại hệ thống chưa có mô hình chi tiêu nào."
        
        # Định dạng phản hồi thân thiện với người dùng
        result = f"📊 **Các mô hình chi tiêu hiện có ({len(models)}):**\n\n"
        
        for idx, model in enumerate(models):
            # Trích xuất thông tin
            name = model.get("name", "Không có tên")
            description = model.get("description", "Không có mô tả")
            
            # Loại bỏ HTML tags từ mô tả
            clean_description = re.sub(r'<[^>]*>', '', description)
            
            # Thêm mô hình vào kết quả
            result += f"**{idx+1}. {name}**\n"
            result += f"{clean_description}\n\n"
        
        result += "Bạn có thể chọn một trong các mô hình chi tiêu trên để áp dụng cho tài chính cá nhân của mình."
        return result
        
    except requests.exceptions.RequestException as e:
        print(f"[TOOLS] API request failed: {str(e)}")
        return f"Đã xảy ra lỗi khi lấy danh sách mô hình chi tiêu: {str(e)}"

def format_currency(amount):
    """Định dạng số tiền theo chuẩn VND."""
    return f"{int(amount):,} VNĐ".replace(",", ".")

def format_date(date_str):
    """Định dạng ngày tháng từ ISO format sang dd/mm/yyyy HH:MM theo giờ Việt Nam."""
    try:
        if not date_str:
            return "Không có ngày"
        
        # Múi giờ Việt Nam (UTC+7)
        vietnam_tz = timezone(timedelta(hours=7))
            
        # Xử lý cả hai định dạng có thể có
        if "T" in date_str:
            # ISO format đầy đủ
            if "." in date_str:
                # Có phần milliseconds
                dt_str = date_str.split(".")[0]
                dt = datetime.strptime(dt_str, "%Y-%m-%dT%H:%M:%S")
            else:
                # Không có phần milliseconds
                dt = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S")
            
            # Giả định là UTC và chuyển về múi giờ Việt Nam
            dt = dt.replace(tzinfo=timezone.utc).astimezone(vietnam_tz)
        else:
            # Chỉ có ngày
            dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=vietnam_tz)
            
        return dt.strftime("%d/%m/%Y %H:%M")
    except Exception as e:
        print(f"[TOOLS] Error formatting date {date_str}: {str(e)}")
        return date_str  # Trả về nguyên bản nếu có lỗi

def format_user_friendly_date(date_str):
    """Định dạng ngày YYYY-MM-DD sang dd/mm/yyyy cho người dùng."""
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return dt.strftime("%d/%m/%Y")
    except Exception:
        return date_str

def format_user_friendly_datetime(datetime_str):
    """Định dạng thời gian ISO YYYY-MM-DDThh:mm:ss sang dd/mm/yyyy hh:mm cho người dùng."""
    try:
        if "T" in datetime_str:
            dt = datetime.strptime(datetime_str, "%Y-%m-%dT%H:%M:%S")
            return dt.strftime("%d/%m/%Y %H:%M")
        else:
            # Nếu không có phần giờ, chỉ có ngày
            dt = datetime.strptime(datetime_str, "%Y-%m-%d")
            return dt.strftime("%d/%m/%Y")
    except Exception as e:
        print(f"[TOOLS] Error formatting datetime {datetime_str}: {str(e)}")
        return datetime_str

tools = [user_input_expense, get_transaction_history, get_current_spending_model, get_available_spending_models]
