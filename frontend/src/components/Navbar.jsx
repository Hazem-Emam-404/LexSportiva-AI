import React from 'react'
import { MessageSquare, BookOpen, Sparkles, Home } from 'lucide-react'

export default function Navbar({ currentView, setView, onNewChat }) {
  return (
    <header className="navbar">
      <div className="nav-brand" onClick={() => setView('landing')}>
        <img
          src="/images/referee_avatar.jpg"
          alt="Sports AI Referee"
          className="nav-brand-img"
        />
        <div className="nav-brand-title">
          Sports AI <span className="text-gradient-cyan">Referee</span>
          <span className="nav-brand-badge">Official</span>
        </div>
      </div>

      <nav className="nav-links">
        <button
          className={`nav-link ${currentView === 'landing' ? 'active' : ''}`}
          onClick={() => setView('landing')}
        >
          <Home size={17} />
          Home
        </button>

        <button
          className={`nav-link ${currentView === 'rulebooks' ? 'active' : ''}`}
          onClick={() => setView('rulebooks')}
        >
          <BookOpen size={17} />
          Rulebooks
        </button>

        <button
          className={`nav-link ${currentView === 'chat' ? 'active' : ''}`}
          onClick={() => setView('chat')}
        >
          <MessageSquare size={17} />
          Chat Assistant
        </button>

        <button
          className="nav-btn-chat"
          onClick={() => {
            onNewChat()
            setView('chat')
          }}
        >
          <Sparkles size={16} />
          New Officiating Chat
        </button>
      </nav>
    </header>
  )
}
