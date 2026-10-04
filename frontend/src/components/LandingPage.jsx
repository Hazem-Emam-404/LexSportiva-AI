import React from 'react'
import { MessageSquare, BookOpen, ShieldCheck, Search, Globe, ChevronRight, FileText } from 'lucide-react'

const SPORTS_DATA = [
  {
    name: 'Football (Soccer)',
    icon: '⚽',
    governingBody: 'IFAB / FIFA',
    fileName: 'Football.pdf',
    description: 'The official Laws of the Game covering fouls, offside, VAR protocols, penalties, and pitch requirements.',
    pages: '215+ pages',
  },
  {
    name: 'Basketball',
    icon: '🏀',
    governingBody: 'FIBA',
    fileName: 'Basketball.pdf',
    description: 'Official Basketball Rules including 24-second shot clock, personal/technical fouls, and court dimensions.',
    pages: '100+ pages',
  },
  {
    name: 'Handball',
    icon: '🤾',
    governingBody: 'IHF',
    fileName: 'Handball.pdf',
    description: 'Rules of the Game covering 7-meter throws, passive play, suspensions, substitutions, and referee hand signals.',
    pages: '90+ pages',
  },
  {
    name: 'Tennis',
    icon: '🎾',
    governingBody: 'ITF',
    fileName: 'Tennis.pdf',
    description: 'ITF Rules of Tennis detailing tie-breaks, serve foot faults, let calls, equipment specifications, and scoring.',
    pages: '50+ pages',
  },
  {
    name: 'Boxing',
    icon: '🥊',
    governingBody: 'IBA / Olympic',
    fileName: 'Boxing.pdf',
    description: 'Technical and Competition Rules governing weight classes, legal blows, knockdowns, fouls, and ringside scoring.',
    pages: '65+ pages',
  },
]

export default function LandingPage({ onStartChat, onOpenPdf }) {
  return (
    <div className="landing-view">
      {/* Hero Section */}
      <section className="hero-section">
        <div>
          <div className="hero-tag">
            <ShieldCheck size={16} />
            Verified Against Official International Rulebooks
          </div>

          <h1 className="hero-title">
            The Official <br />
            <span className="text-gradient-cyan">Sports Rules</span> <br />
            AI Referee
          </h1>

          <p className="hero-description">
            Get instant, authoritative answers to complex sports officiating questions.
            Every referee ruling is grounded in official rulebooks with exact page citations.
          </p>

          <div className="hero-actions">
            <button className="btn-primary" onClick={onStartChat}>
              <MessageSquare size={19} />
              Start Officiating Chat
            </button>
            <a href="#rulebooks" className="btn-secondary">
              <BookOpen size={19} />
              Browse Rulebooks
            </a>
          </div>
        </div>

        {/* Hero Avatar Presentation */}
        <div className="hero-avatar-wrapper">
          <div className="hero-avatar-card">
            <img
              src="/images/referee_avatar.jpg"
              alt="AI Referee Official Character"
              className="hero-avatar-img"
            />
            <div className="hero-avatar-badge">
              <div>
                <div style={{ fontSize: '0.95rem', fontWeight: 800 }}>AI Referee 01</div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-sub)' }}>Hybrid RAG Assistant</div>
              </div>
              <div className="avatar-badge-status">
                <span className="status-dot"></span>
                Active
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Feature Pillars */}
      <section style={{ marginBottom: '6rem' }}>
        <div className="section-header">
          <div className="section-tag">Key Capabilities</div>
          <h2 className="section-title">Built for Referees, Coaches & Fans</h2>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1.5rem' }}>
          <div className="glass-panel" style={{ padding: '2rem', borderRadius: 'var(--radius-lg)' }}>
            <div style={{ color: 'var(--accent-cyan)', marginBottom: '1rem' }}>
              <FileText size={32} />
            </div>
            <h3 style={{ fontSize: '1.25rem', marginBottom: '0.5rem' }}>Verifiable Page Citations</h3>
            <p style={{ color: 'var(--text-sub)', fontSize: '0.95rem' }}>
              Every claim quotes the exact rulebook name and page number. Tap any citation to inspect the official PDF directly at that page.
            </p>
          </div>

          <div className="glass-panel" style={{ padding: '2rem', borderRadius: 'var(--radius-lg)' }}>
            <div style={{ color: 'var(--accent-gold)', marginBottom: '1rem' }}>
              <Search size={32} />
            </div>
            <h3 style={{ fontSize: '1.25rem', marginBottom: '0.5rem' }}>Hybrid Multi-Sport Search</h3>
            <p style={{ color: 'var(--text-sub)', fontSize: '0.95rem' }}>
              Combines dense BGE-M3 embeddings, BM25 keyword matching, cross-encoder reranking, and parent-child document resolution.
            </p>
          </div>

          <div className="glass-panel" style={{ padding: '2rem', borderRadius: 'var(--radius-lg)' }}>
            <div style={{ color: 'var(--accent-emerald)', marginBottom: '1rem' }}>
              <Globe size={32} />
            </div>
            <h3 style={{ fontSize: '1.25rem', marginBottom: '0.5rem' }}>Bilingual (English & Arabic)</h3>
            <p style={{ color: 'var(--text-sub)', fontSize: '0.95rem' }}>
              Ask in Arabic or English with seamless query translation, cross-lingual retrieval, and natural language explanations.
            </p>
          </div>
        </div>
      </section>

      {/* Rulebooks Section */}
      <section id="rulebooks">
        <div className="section-header">
          <div className="section-tag">Library</div>
          <h2 className="section-title">Official Governed Rulebooks</h2>
          <p style={{ color: 'var(--text-sub)', maxWidth: '600px', margin: '0 auto' }}>
            Click any rulebook below to view and read the official federation document directly in your browser.
          </p>
        </div>

        <div className="rulebooks-grid">
          {SPORTS_DATA.map((sport) => (
            <div
              key={sport.name}
              className="sport-card"
              onClick={() => onOpenPdf(sport.fileName, 1)}
            >
              <div>
                <div className="sport-card-top">
                  <div className="sport-icon-box">{sport.icon}</div>
                  <span className="sport-pages-badge">{sport.governingBody}</span>
                </div>
                <h3 className="sport-name">{sport.name}</h3>
                <p className="sport-rulebook-title">{sport.description}</p>
              </div>

              <div>
                <button
                  className="sport-btn-open"
                  onClick={(e) => {
                    e.stopPropagation()
                    onOpenPdf(sport.fileName, 1)
                  }}
                >
                  <FileText size={15} />
                  Open PDF ({sport.pages})
                  <ChevronRight size={15} />
                </button>
              </div>
            </div>
          ))}
        </div>
      </section>
    </div>
  )
}
