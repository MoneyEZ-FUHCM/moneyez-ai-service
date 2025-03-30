from typing import List
import json

from fastapi import FastAPI, Request
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    ToolMessage,
    SystemMessage,
    BaseMessage,
)

from app.exception.exception import CustomHTTPException

from .models import (
    BaseResponse,
    LanguageModelV1Message,
    LanguageModelTextPart,
    LanguageModelImagePart,
    LanguageModelToolCallPart,
    LanguageModelUserMessage,
    ChatRequest,
    MessageResponse,
    RevicedMessage,
    RevicedMessageParse,
    MessageBaseCShap
)
from fastapi import status

# Global variable for user ID
userId = None

def convert_to_langchain_messages(
        messages: List[LanguageModelV1Message],
) -> List[BaseMessage]:
    result = []

    for msg in messages:
        if msg.role == "system":
            result.append(SystemMessage(content=msg.content))

        elif msg.role == "user":
            content = []
            for p in msg.content:
                if isinstance(p, LanguageModelTextPart):
                    content.append({"type": "text", "text": p.text})
                elif isinstance(p, LanguageModelImagePart):
                    content.append({"type": "image_url", "image_url": p.image})
            result.append(HumanMessage(content=content))

        elif msg.role == "assistant":
            text_parts = [
                p for p in msg.content if isinstance(p, LanguageModelTextPart)
            ]
            text_content = " ".join(p.text for p in text_parts)
            tool_calls = [
                {
                    "id": p.toolCallId,
                    "name": p.toolName,
                    "args": p.args,
                }
                for p in msg.content
                if isinstance(p, LanguageModelToolCallPart)
            ]
            result.append(AIMessage(content=text_content, tool_calls=tool_calls))

        elif msg.role == "tool":
            for tool_result in msg.content:
                result.append(
                    ToolMessage(
                        content=str(tool_result.result),
                        tool_call_id=tool_result.toolCallId,
                    )
                )

    return result

def add_langgraph_route(app: FastAPI, graph, base_path: str):
    
    @app.post("/api/receive_message", response_model=BaseResponse)
    async def receive_message_endpoint(
        request_request: Request,
        revicedMessage: RevicedMessage
    ):
        # Validate X-External-Secret header"
        
        if "X-External-Secret" not in request_request.headers:
            print("[CHAT] Missing X-External-Secret header")
            raise CustomHTTPException(
                status_code=403,
                error_code="UNAUTHORIZED",
                message="Missing X-External-Secret header"
            )
        
        if request_request.headers["X-External-Secret"] != "thisIsSerectKeyPythonService":
            print("[CHAT] Invalid X-External-Secret header")
            raise CustomHTTPException(
                status_code=403,
                error_code="UNAUTHORIZED",
                message="Invalid X-External-Secret header"
            )
        try:
            # Parse the received data
            parsed_data = json.loads(revicedMessage.data)
            
            # Convert to RevicedMessageParse object
            message_obj = RevicedMessageParse(
                userId=parsed_data.get("UserId"),
                message=parsed_data.get("Message"),
                conversationId=parsed_data.get("ConversationId"),
                PreviousMessages=[]
            )
            
            # Parse the previous messages
            if "PreviousMessages" in parsed_data and parsed_data["PreviousMessages"]:
                for msg in parsed_data["PreviousMessages"]:
                    message_obj.PreviousMessages.append(
                        MessageBaseCShap(
                            ConversationId=msg.get("ConversationId"),
                            Content=msg.get("Content"),
                            Role=msg.get("Role"),
                            Timestamp=msg.get("Timestamp")
                        )
                    )
            
            print(f"[CHAT] Message type: {type(message_obj)}")
            global userId
            userId = message_obj.userId
            conversation_id = message_obj.conversationId
            
            print(f"[CHAT] Received message from user {userId} in conversation {conversation_id}")
            print(f"[CHAT] Message content: {message_obj.message}")
        except Exception as e:
            print(f"[CHAT] Error parsing message: {str(e)}")
            raise CustomHTTPException(
                status_code=400,
                error_code="INVALID_REQUEST",
                message=f"Invalid request format: {str(e)}"
            )
        
        
        # Get previous messages if available
        previous_messages = message_obj.PreviousMessages
        print(f"[CHAT] Previous message count: {len(previous_messages) if previous_messages else 0}")
        
        # Log all previous messages
        if previous_messages:
            print("[CHAT] Previous messages in conversation:")
            for i, msg in enumerate(previous_messages):
                print(f"[CHAT] [History {i}] Role: {msg.Role}, Content: {msg.Content[:100]}...")
        
        # Create new message
        new_message = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": message_obj.message
                }
            ]
        }
        
        # Create a LanguageModelUserMessage object from the message
        user_message_obj = LanguageModelUserMessage(**new_message)
        
        # Create ChatRequest with the LanguageModelUserMessage
        chat_request = ChatRequest(
            system="",
            tools=[],
            messages=[user_message_obj]
        )
        
        # Convert messages to proper format
        new_messages = convert_to_langchain_messages(chat_request.messages)
        
        # Convert previous messages to the correct format for the model
        previous_langchain_messages = []
        if previous_messages:
            for msg in previous_messages:
                if msg.Role.upper() == "USER":
                    previous_langchain_messages.append(HumanMessage(content=msg.Content))
                elif msg.Role.upper() == "BOT" or msg.Role.upper() == "ASSISTANT":
                    previous_langchain_messages.append(AIMessage(content=msg.Content))
                elif msg.Role.upper() == "SYSTEM":
                    previous_langchain_messages.append(SystemMessage(content=msg.Content))
        
        # Combine with previous messages if available
        if previous_langchain_messages and len(new_messages) == 1 and isinstance(new_messages[0], HumanMessage):
            inputs = previous_langchain_messages + new_messages
            print(f"[CHAT] Initializing with {len(inputs)} messages ({len(previous_langchain_messages)} from history)")
        else:
            inputs = new_messages
            print(f"[CHAT] Using {len(inputs)} messages from request")
        
        # Log the complete conversation history being sent to the model
        print("[CHAT] Complete conversation history:")
        for i, msg in enumerate(inputs):
            msg_type = type(msg).__name__
            msg_content = msg.content
            if isinstance(msg_content, list):
                msg_content = str(msg_content)[:100] + "..."
            elif isinstance(msg_content, str) and len(msg_content) > 100:
                msg_content = msg_content[:100] + "..."
            print(f"[CHAT] [{i}] Type: {msg_type}, Content: {msg_content}")
        
        # Create config for the graph
        config = {
            "configurable": {
                "system": chat_request.system,
                "frontend_tools": chat_request.tools,
                "thread_id": conversation_id
            }
        }
        
        print(f"[CHAT] Invoking graph with config: {config}")
        
        # Call graph.invoke to get results immediately
        try:
            print(f"[CHAT] Sending request to AI model")
            response = await graph.ainvoke({"messages": inputs}, config)
            print(f"[CHAT] Received response from AI model")
            
            # Get the last message in the response (typically the AI's response)
            ai_message = None
            for msg in response.get("messages", []):
                if isinstance(msg, AIMessage):
                    ai_message = msg
            
            if ai_message:
                print(f"[CHAT] AI response content: {ai_message.content[:100]}...")
                
                # Format the message to return
                formatted_response = {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": ai_message.content
                        }
                    ]
                }
                
                response = MessageResponse(
                    status="success",
                    conversation_id=conversation_id,
                    message=formatted_response
                )

                print(f"[CHAT] Successfully generated response for conversation {conversation_id}")
                return BaseResponse(
                    status=status.HTTP_200_OK,
                    message="Response generated successfully",
                    data=response
                )
            else:
                print(f"[CHAT] No assistant message found in response")
                return BaseResponse(
                    status=status.HTTP_400_BAD_REQUEST,
                    message="No assistant message found in response",
                    data=MessageResponse(
                        status="error",
                        conversation_id=conversation_id,
                        message="No response generated"
                    )
                )
            
        except Exception as e:
            print(f"[CHAT] Error in receive_message_endpoint: {str(e)}")
            return BaseResponse(
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                message=f"Error generating response: {str(e)}",
                data=MessageResponse(
                    status="error",
                    conversation_id=conversation_id,
                    message=str(e)
                )
            )

    # app.add_api_route(
    #     "/api/receive_message",
    #     receive_message_endpoint,
    #     methods=["POST"],
    #     response_model=BaseResponse,
    #     tags=["LangGraph"]
    # )
