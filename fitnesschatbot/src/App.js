import React, { useState } from 'react';
import './chat.css'; // Your original CSS file
 // Import the new full-screen CSS file
import Axios from 'axios';
import './App.css';

function Chat() {
  const [userInput, setUserInput] = useState('');
  const [chatHistory, setChatHistory] = useState([]);

  const handleInputChange = (e) => {
    setUserInput(e.target.value);
  };

  const handleSendMessage = async () => {
    try {
      const response = await Axios.post('http://127.0.0.1:5000/api/send-message', { user_input: userInput });
      const botResponse = response.data.bot_response;
      setChatHistory([...chatHistory, `You: ${userInput}`, `Chatbot: ${botResponse}`]);
      setUserInput('');
    } catch (error) {
      console.error('Error sending message:', error);
    }
  };

  return (
    <div className="chat-container">
      <h1>Fitness Chatbot</h1> {/* Added heading here */}
      <div className="chat-history">
        {chatHistory.map((message, index) => (
          <div key={index} className="chat-message">
            {message}
          </div>
        ))}
      </div>
      <div className="chat-input">
        <input
          type="text"
          placeholder="Type your message..."
          value={userInput}
          onChange={handleInputChange}
        />
        <button onClick={handleSendMessage}>Send</button>
      </div>
    </div>
  );
}

export default Chat;
