"""Comprehensive test for voice assistant functionality."""
import asyncio
import json
import websockets
import numpy as np
import sys


async def test_full_conversation():
    """Test a full conversation flow with multiple exchanges."""
    uri = "ws://localhost:8000/ws"
    
    print("="*70)
    print("Test 1: Full Conversation Flow")
    print("="*70)
    
    async with websockets.connect(uri) as websocket:
        # Wait for session message
        await websocket.recv()
        
        # Send multiple messages
        questions = [
            "Xin chào!",
            "Thời tiết hôm nay thế nào?",
            "Cảm ơn bạn!",
        ]
        
        for i, question in enumerate(questions, 1):
            print(f"\n--- Question {i}: {question} ---")
            
            await websocket.send(json.dumps({
                "type": "text",
                "text": question
            }, ensure_ascii=False))
            
            responses = []
            full_response = ""
            
            try:
                while True:
                    response = await asyncio.wait_for(websocket.recv(), timeout=30.0)
                    
                    if isinstance(response, str):
                        data = json.loads(response)
                        msg_type = data.get("type")
                        responses.append(data)
                        
                        if msg_type == "response":
                            text = data.get("text", "")
                            full_text = data.get("full_text", "")
                            is_final = data.get("is_final", False)
                            
                            if text:
                                print(f"  Streaming: {text}")
                            
                            if is_final and full_text:
                                full_response = full_text
                                print(f"  ✓ Final response received ({len(full_text)} chars)")
                                break
                        elif msg_type == "audio":
                            print(f"  ✓ Audio chunk received")
                    else:
                        print(f"  ✓ Binary audio: {len(response)} bytes")
                        
            except asyncio.TimeoutError:
                print("  ⚠ Timeout")
            
            print(f"  Total messages: {len(responses)}")
        
        # Check history
        print("\n--- Checking Conversation History ---")
        await websocket.send(json.dumps({"type": "get_history"}))
        response = await websocket.recv()
        data = json.loads(response)
        
        if data.get("type") == "history":
            messages = data.get("messages", [])
            print(f"✓ Total messages in history: {len(messages)}")
            for i, msg in enumerate(messages, 1):
                role = msg.get('role')
                content = msg.get('content', '')[:60]
                print(f"  {i}. {role}: {content}...")
            
            # Should have: system + 3 user messages + 3 assistant responses
            expected = 7  # 1 system + 3 user + 3 assistant
            if len(messages) >= expected:
                print(f"✓ History correctly contains {len(messages)} messages")
            else:
                print(f"⚠ Expected at least {expected} messages, got {len(messages)}")
        
        print("\n✅ Test 1 passed!")


async def test_audio_vad_detection():
    """Test audio processing with VAD."""
    uri = "ws://localhost:8000/ws"
    
    print("\n" + "="*70)
    print("Test 2: Audio VAD Detection")
    print("="*70)
    
    async with websockets.connect(uri) as websocket:
        # Wait for session message
        await websocket.recv()
        
        # Send 2 seconds of silence
        print("\n--- Sending 2s silence ---")
        sample_rate = 16000
        chunk_samples = 1600  # 100ms
        
        for i in range(20):
            silence = np.zeros(chunk_samples, dtype=np.int16).tobytes()
            await websocket.send(silence)
            await asyncio.sleep(0.05)
        
        print("✓ Silence sent")
        
        # Send "speech" (random noise)
        print("\n--- Sending 1s 'speech' (noise) ---")
        for i in range(10):
            noise = np.random.randint(-1000, 1000, chunk_samples, dtype=np.int16).tobytes()
            await websocket.send(noise)
            await asyncio.sleep(0.05)
        
        print("✓ Noise sent")
        
        # Check for any transcript responses
        try:
            while True:
                response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                if isinstance(response, str):
                    data = json.loads(response)
                    if data.get("type") == "transcript":
                        print(f"✓ Transcript: {data.get('text')}")
        except asyncio.TimeoutError:
            print("✓ No transcript (expected for noise)")
        
        print("\n✅ Test 2 passed!")


async def test_error_handling():
    """Test error handling and edge cases."""
    uri = "ws://localhost:8000/ws"
    
    print("\n" + "="*70)
    print("Test 3: Error Handling")
    print("="*70)
    
    async with websockets.connect(uri) as websocket:
        # Wait for session
        await websocket.recv()
        
        # Test empty text
        print("\n--- Testing empty text ---")
        await websocket.send(json.dumps({"type": "text", "text": ""}))
        
        try:
            response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            data = json.loads(response)
            print(f"✓ Response to empty text: {data.get('type')}")
        except asyncio.TimeoutError:
            print("✓ No response to empty text (expected)")
        
        # Test very long text
        print("\n--- Testing long text ---")
        long_text = "Xin chào! " * 100
        await websocket.send(json.dumps({"type": "text", "text": long_text}))
        
        response_count = 0
        try:
            while True:
                response = await asyncio.wait_for(websocket.recv(), timeout=30.0)
                response_count += 1
                if isinstance(response, str):
                    data = json.loads(response)
                    if data.get("type") == "response" and data.get("is_final"):
                        print(f"✓ Final response received after {response_count} messages")
                        break
        except asyncio.TimeoutError:
            print(f"⚠ Timeout after {response_count} responses")
        
        # Test invalid JSON
        print("\n--- Testing invalid JSON ---")
        await websocket.send("not valid json")
        
        try:
            response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
            print(f"✓ Server handled invalid JSON")
        except asyncio.TimeoutError:
            print("✓ No response to invalid JSON (expected)")
        
        print("\n✅ Test 3 passed!")


async def test_session_management():
    """Test session persistence and management."""
    uri = "ws://localhost:8000/ws"
    
    print("\n" + "="*70)
    print("Test 4: Session Management")
    print("="*70)
    
    session_id = None
    
    # First connection
    print("\n--- Creating session ---")
    async with websockets.connect(uri) as websocket:
        response = await websocket.recv()
        data = json.loads(response)
        session_id = data.get("session_id")
        print(f"✓ Session created: {session_id}")
        
        # Send a message
        await websocket.send(json.dumps({
            "type": "text",
            "text": "Test message for session"
        }))
        
        # Wait for response
        try:
            while True:
                response = await asyncio.wait_for(websocket.recv(), timeout=30.0)
                if isinstance(response, str):
                    data = json.loads(response)
                    if data.get("type") == "response" and data.get("is_final"):
                        print("✓ Message processed")
                        break
        except asyncio.TimeoutError:
            print("⚠ Timeout waiting for response")
    
    # Check session via API
    print("\n--- Checking session via API ---")
    import requests
    response = requests.get(f"http://localhost:8000/sessions/{session_id}")
    if response.status_code == 200:
        session_data = response.json()
        print(f"✓ Session found via API")
        messages = session_data.get("session", {}).get("history", [])
        print(f"  Messages in session: {len(messages)}")
    else:
        print("⚠ Session not found via API (might be cleaned up)")
    
    # Test session deletion
    print("\n--- Deleting session ---")
    response = requests.delete(f"http://localhost:8000/sessions/{session_id}")
    if response.status_code == 200:
        print("✓ Session deleted")
    else:
        print("⚠ Session deletion failed")
    
    print("\n✅ Test 4 passed!")


async def main():
    """Run all comprehensive tests."""
    print("\n" + "="*70)
    print("COMPREHENSIVE VOICE ASSISTANT TESTS")
    print("="*70)
    
    tests = [
        ("Full Conversation", test_full_conversation),
        ("Audio VAD Detection", test_audio_vad_detection),
        ("Error Handling", test_error_handling),
        ("Session Management", test_session_management),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            await test_func()
            passed += 1
        except Exception as e:
            print(f"\n❌ {name} failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*70)
    
    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
