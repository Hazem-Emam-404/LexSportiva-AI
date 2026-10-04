import React, { useState, useEffect, useRef } from 'react'
import {
  Plus,
  Trash2,
  Send,
  PanelLeftClose,
  PanelLeft,
  BookOpen,
  Sparkles,
  ArrowUp
} from 'lucide-react'
import {
  fetchConversations,
  getConversation,
  deleteConversation,
  sendMessage,
} from '../services/api'

const QUICK_PROMPTS = [
  'What is the offside rule in football?',
  'How many personal fouls before a player is disqualified in basketball?',
  'What is considered passive play in handball?',
  'What is a let call during a tennis serve?',
  'What are the mandatory standing 8-count rules in boxing?',
  'ما هو حجم ملعب كرة القدم القانوني؟',
]

export default function ChatView({
  activeConvId,
  setActiveConvId,
  onOpenCitations,
}) {
  const [conversations, setConversations] = useState([])
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const messagesEndRef = useRef(null)
  const textareaRef = useRef(null)

  // Load conversations on mount
  useEffect(() => {
    loadConversations()
  }, [])

  // When activeConvId changes, load messages (if activeConvId is set)
  useEffect(() => {
    if (activeConvId) {
      loadMessages(activeConvId)
    } else {
      setMessages([])
    }
  }, [activeConvId])

  // Scroll to bottom on new message
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isLoading])

  const loadConversations = async () => {
    try {
      const data = await fetchConversations()
      setConversations(data)
    } catch (err) {
      console.error('Failed to load conversations:', err)
    }
  }

  const loadMessages = async (convId) => {
    try {
      setIsLoading(true)
      const data = await getConversation(convId)
      setMessages(data.messages || [])
    } catch (err) {
      console.error('Failed to load messages for conversation:', err)
    } finally {
      setIsLoading(false)
    }
  }

  const handleNewChat = () => {
    setActiveConvId(null)
    setMessages([])
    setInput('')
    if (textareaRef.current) {
      textareaRef.current.focus()
    }
  }

  const handleDeleteConversation = async (e, convId) => {
    e.stopPropagation()
    try {
      await deleteConversation(convId)
      setConversations((prev) => prev.filter((c) => c.id !== convId))
      if (activeConvId === convId) {
        handleNewChat()
      }
    } catch (err) {
      console.error('Failed to delete conversation:', err)
    }
  }

  const handleSend = async (textToSend = null) => {
    const query = (textToSend || input).trim()
    if (!query || isLoading) return

    // Optimistically append user message
    const tempUserMsg = {
      id: `temp_user_${Date.now()}`,
      role: 'user',
      content: query,
      created_at: new Date().toISOString(),
    }
    setMessages((prev) => [...prev, tempUserMsg])
    setInput('')
    setIsLoading(true)

    try {
      // Backend automatically creates a conversation if activeConvId is null
      const res = await sendMessage(query, activeConvId)

      const assistantMsg = {
        id: `assistant_${Date.now()}`,
        role: 'assistant',
        content: res.answer,
        citations: res.citations || [],
        created_at: new Date().toISOString(),
      }

      setMessages((prev) => [...prev, assistantMsg])

      // If this was a new conversation, update activeConvId and reload conversation list
      if (!activeConvId && res.conversation_id) {
        setActiveConvId(res.conversation_id)
        await loadConversations()
      }
    } catch (err) {
      console.error('Chat error:', err)
      const errorMsg = {
        id: `err_${Date.now()}`,
        role: 'assistant',
        content: `⚠️ Error: ${err.message || 'Unable to get ruling from the referee. Please try again.'}`,
        citations: [],
        created_at: new Date().toISOString(),
      }
      setMessages((prev) => [...prev, errorMsg])
    } finally {
      setIsLoading(false)
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div className="chat-layout">
      {/* Gemini Style Left Sidebar */}
      <aside className={`chat-sidebar ${sidebarOpen ? '' : 'collapsed'}`}>
        <div className="sidebar-header">
          <button className="btn-new-chat" onClick={handleNewChat}>
            <Plus size={18} style={{ color: 'var(--accent-cyan)' }} />
            New Officiating Chat
          </button>
          <div className="sidebar-label">Recent Inquiries</div>
        </div>

        <div className="conversations-list">
          {conversations.length === 0 ? (
            <div style={{ padding: '1rem', color: 'var(--text-dim)', fontSize: '0.85rem' }}>
              No previous chats found.
            </div>
          ) : (
            conversations.map((c) => (
              <div
                key={c.id}
                className={`conversation-item ${activeConvId === c.id ? 'active' : ''}`}
                onClick={() => setActiveConvId(c.id)}
              >
                <span className="conv-title" title={c.title}>
                  {c.title || 'Untitled Query'}
                </span>
                <button
                  className="btn-delete-conv"
                  onClick={(e) => handleDeleteConversation(e, c.id)}
                  title="Delete Chat"
                >
                  <Trash2 size={15} />
                </button>
              </div>
            ))
          )}
        </div>
      </aside>

      {/* Main Chat Area */}
      <main className="chat-main">
        {/* Header */}
        <div className="chat-header">
          <div className="chat-header-left">
            <button
              className="btn-toggle-sidebar"
              onClick={() => setSidebarOpen((prev) => !prev)}
              title={sidebarOpen ? 'Collapse Sidebar' : 'Expand Sidebar'}
            >
              {sidebarOpen ? <PanelLeftClose size={18} /> : <PanelLeft size={18} />}
            </button>

            <span className="chat-active-title">
              {activeConvId
                ? conversations.find((c) => c.id === activeConvId)?.title || 'LexSportiva Chat'
                : 'New Officiating Consultation'}
            </span>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.8rem', color: 'var(--accent-cyan)' }}>
            <span className="status-dot"></span>
            Hybrid RAG Active
          </div>
        </div>

        {/* Message Thread */}
        <div className="chat-messages">
          {messages.length === 0 ? (
            <div className="chat-welcome">
              <img
                src="/images/referee_avatar.jpg"
                alt="AI Referee"
                className="welcome-avatar"
              />
              <h2 className="welcome-title">What rule can I clarify for you?</h2>
              <p className="welcome-subtitle">
                Official rulebook grounding across Football, Basketball, Handball, Tennis, and Boxing.
              </p>

              <div className="welcome-chips-label">Quick Officiating Questions:</div>
              <div className="welcome-chips">
                {QUICK_PROMPTS.map((prompt, idx) => (
                  <button
                    key={idx}
                    className="prompt-chip"
                    onClick={() => handleSend(prompt)}
                  >
                    <span>{prompt}</span>
                    <Sparkles size={15} style={{ color: 'var(--accent-cyan)', opacity: 0.7 }} />
                  </button>
                ))}
              </div>
            </div>
          ) : (
            messages.map((m) => (
              <div key={m.id} className={`message-row ${m.role}`}>
                {m.role === 'assistant' && (
                  <img
                    src="/images/referee_avatar.jpg"
                    alt="AI Referee"
                    className="message-avatar"
                  />
                )}

                <div className="message-content-wrapper">
                  <div className="message-bubble">{m.content}</div>

                  {/* Show Citations Button if citations exist */}
                  {m.role === 'assistant' && m.citations && m.citations.length > 0 && (
                    <button
                      className="btn-view-citations"
                      onClick={() => onOpenCitations(m.citations)}
                    >
                      <BookOpen size={14} />
                      View {m.citations.length} Official Citations
                    </button>
                  )}
                </div>

                {m.role === 'user' && (
                  <div className="user-avatar-placeholder">You</div>
                )}
              </div>
            ))
          )}

          {isLoading && (
            <div className="message-row assistant">
              <img
                src="/images/referee_avatar.jpg"
                alt="AI Referee"
                className="message-avatar"
              />
              <div className="message-content-wrapper">
                <div className="message-bubble" style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                  <div className="typing-indicator">
                    <span className="typing-dot"></span>
                    <span className="typing-dot"></span>
                    <span className="typing-dot"></span>
                  </div>
                  <span style={{ fontSize: '0.88rem', color: 'var(--text-sub)' }}>
                    Referee is reviewing official rulebooks...
                  </span>
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Input Bar */}
        <div className="chat-input-container">
          <div className="chat-input-box">
            <textarea
              ref={textareaRef}
              className="chat-textarea"
              placeholder="Ask an officiating or rule question (e.g. Football, Basketball, Tennis, Boxing)..."
              value={input}
              rows={1}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={isLoading}
            />
            <button
              className="btn-send"
              onClick={() => handleSend()}
              disabled={!input.trim() || isLoading}
              title="Send Message"
            >
              <ArrowUp size={18} />
            </button>
          </div>
        </div>
      </main>
    </div>
  )
}
