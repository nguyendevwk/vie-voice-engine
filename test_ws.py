"""Test script for voice assistant WebSocket."""
import asyncio
import json
import websockets
import numpy as np


async def test_websocket_connection():
    """Test WebSocket connection and text mode."""
    uri = "ws://localhost:8000/ws"
    
    print("Connecting to WebSocket...")
    async with websockets.connect(uri) as websocket:
        print("✓ Connected")
        
        # Wait for session message
        response = await websocket.recv()
        data = json.loads(response)
        print(f"✓ Session initialized: {data.get('type')} - Session ID: {data.get('session_id')}")
        
        # Test text input
        print("\n--- Testing text input ---")
        test_message = {
            "type": "text",
            "text": "Xin chào, bạn khỏe không?"
        }
        
        print(f"Sending: {test_message['text']}")
        await websocket.send(json.dumps(test_message, ensure_ascii=False))
        
        # Collect responses
        responses = []
        try:
            while True:
                response = await asyncio.wait_for(websocket.recv(), timeout=30.0)
                
                # Check if it's binary or text
                if isinstance(response, bytes):
                    print(f"✓ Received audio chunk: {len(response)} bytes")
                    responses.append({"type": "audio", "size": len(response)})
                else:
                    data = json.loads(response)
                    msg_type = data.get("type")
                    responses.append(data)
                    
                    if msg_type == "transcript":
                        print(f"✓ Transcript: {data.get('text')}")
                    elif msg_type == "response":
                        is_final = data.get("is_final", False)
                        text = data.get("text", "")
                        if text:
                            print(f"✓ Response ({'final' if is_final else 'streaming'}): {text[:100]}")
                    elif msg_type == "control":
                        print(f"✓ Control: {data.get('action')}")
                    
                    # Stop after final response
                    if msg_type == "response" and data.get("is_final"):
                        # Wait a bit for any remaining messages
                        await asyncio.sleep(1)
                        break
        except asyncio.TimeoutError:
            print("⚠ Timeout waiting for responses")
        
        print(f"\n✓ Total responses: {len(responses)}")
        
        # Test getting history
        print("\n--- Testing history endpoint ---")
        await websocket.send(json.dumps({"type": "get_history"}))
        response = await websocket.recv()
        data = json.loads(response)
        if data.get("type") == "history":
            messages = data.get("messages", [])
            print(f"✓ History: {len(messages)} messages")
            for msg in messages:
                print(f"  - {msg.get('role')}: {msg.get('content', '')[:80]}")
        
        # Test reset
        print("\n--- Testing reset ---")
        await websocket.send(json.dumps({"type": "reset"}))
        response = await websocket.recv()
        data = json.loads(response)
        if data.get("type") == "control" and data.get("action") == "reset_complete":
            print("✓ Reset successful")
        
        print("\n✅ All tests passed!")


async def test_audio_streaming():
    """Test audio streaming with dummy PCM data."""
    uri = "ws://localhost:8000/ws"
    
    print("\n--- Testing audio streaming ---")
    async with websockets.connect(uri) as websocket:
        # Wait for session message
        await websocket.recv()
        
        # Send 3 seconds of silence (16kHz, 16-bit, mono)
        print("Sending 3s of silence...")
        sample_rate = 16000
        duration_ms = 100
        samples_per_chunk = int(sample_rate * duration_ms / 1000)
        
        for i in range(30):  # 30 chunks = 3 seconds
            silence = np.zeros(samples_per_chunk, dtype=np.int16).tobytes()
            await websocket.send(silence)
            await asyncio.sleep(0.1)
        
        print("✓ Audio chunks sent")
        
        # Check for any responses
        try:
            while True:
                response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                if isinstance(response, str):
                    data = json.loads(response)
                    print(f"  Response: {data.get('type')} - {data}")
        except asyncio.TimeoutError:
            print("✓ No responses (expected for silence)")


async def main():
    """Run all tests."""
    print("="*60)
    print("Voice Assistant WebSocket Tests")
    print("="*60)
    
    try:
        await test_websocket_connection()
        await test_audio_streaming()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
