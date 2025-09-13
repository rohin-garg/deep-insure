import React from 'react';
import { ChatBar } from '../common/ChatBar';
import { Header } from '../common/Header';
import { ChatComponent } from './ChatComponent';
import { SourceComponent } from './SourceComponent';

function ChatPage() {
  return (
    <div>
      <Header></Header>
      <ChatComponent></ChatComponent>
      <SourceComponent></SourceComponent>
      <ChatBar></ChatBar>
    </div>
  );
}

export default ChatPage;
