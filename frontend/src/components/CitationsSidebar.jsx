import React from 'react'
import { X, BookOpen, ExternalLink, FileText, CheckCircle2 } from 'lucide-react'

export default function CitationsSidebar({ isOpen, onClose, citations, onOpenPdf }) {
  if (!isOpen) return null

  return (
    <>
      {/* Backdrop */}
      <div
        className={`citations-drawer-overlay ${isOpen ? 'open' : ''}`}
        onClick={onClose}
      />

      {/* Drawer */}
      <aside className={`citations-drawer ${isOpen ? 'open' : ''}`}>
        <div className="citations-drawer-header">
          <div className="citations-drawer-title">
            <BookOpen size={20} style={{ color: 'var(--accent-cyan)' }} />
            Official Citations ({citations?.length || 0})
          </div>
          <button className="btn-close-drawer" onClick={onClose} title="Close Citations">
            <X size={20} />
          </button>
        </div>

        <div className="citations-list">
          {(!citations || citations.length === 0) ? (
            <div style={{ textAlign: 'center', color: 'var(--text-dim)', marginTop: '3rem' }}>
              No rulebook citations attached to this response.
            </div>
          ) : (
            citations.map((c, index) => (
              <div key={index} className="citation-card">
                <div className="citation-card-header">
                  <span className="citation-sport-tag">{c.sport || 'Rulebook'}</span>
                  <span className="citation-page-tag">
                    <FileText size={14} />
                    Page {c.page_number}
                  </span>
                </div>

                <div className="citation-claim-quote">
                  "{c.claim}"
                </div>

                <div style={{ fontSize: '0.8rem', color: 'var(--text-sub)' }}>
                  Source: <strong>{c.file_name}</strong>
                </div>

                <button
                  className="btn-open-pdf-page"
                  onClick={() => onOpenPdf(c.file_name, c.page_number)}
                >
                  <ExternalLink size={15} />
                  Open PDF at Page {c.page_number}
                </button>
              </div>
            ))
          )}
        </div>
      </aside>
    </>
  )
}
