import React from 'react'
import { X, ExternalLink, FileText } from 'lucide-react'
import { getPdfUrl } from '../services/api'

export default function PdfViewerModal({ isOpen, onClose, pdfFile, pageNumber }) {
  if (!isOpen || !pdfFile) return null

  const pdfUrl = getPdfUrl(pdfFile, pageNumber)

  return (
    <div className="pdf-modal-backdrop" onClick={onClose}>
      <div className="pdf-modal-content" onClick={(e) => e.stopPropagation()}>
        <div className="pdf-modal-header">
          <div className="pdf-modal-title">
            <FileText size={18} style={{ color: 'var(--accent-cyan)' }} />
            <span>{pdfFile}</span>
            {pageNumber && (
              <span style={{ color: 'var(--accent-gold)', fontSize: '0.85rem' }}>
                (Page {pageNumber})
              </span>
            )}
          </div>

          <div className="pdf-modal-actions">
            <a
              href={pdfUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="btn-secondary"
              style={{ padding: '0.35rem 0.85rem', fontSize: '0.85rem' }}
              title="Open full PDF in new tab"
            >
              <ExternalLink size={14} />
              Open in Tab
            </a>
            <button
              className="btn-close-drawer"
              onClick={onClose}
              title="Close PDF Viewer"
            >
              <X size={20} />
            </button>
          </div>
        </div>

        <iframe
          src={pdfUrl}
          title={`PDF Preview: ${pdfFile}`}
          className="pdf-modal-frame"
        />
      </div>
    </div>
  )
}
