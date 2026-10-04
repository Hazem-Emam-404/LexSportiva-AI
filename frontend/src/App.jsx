import React, { useState } from 'react'
import Navbar from './components/Navbar'
import LandingPage from './components/LandingPage'
import ChatView from './components/ChatView'
import CitationsSidebar from './components/CitationsSidebar'
import PdfViewerModal from './components/PdfViewerModal'

export default function App() {
  const [currentView, setView] = useState('landing') // 'landing' | 'chat' | 'rulebooks'
  const [activeConvId, setActiveConvId] = useState(null)

  // Citations Drawer State
  const [isCitationsOpen, setIsCitationsOpen] = useState(false)
  const [activeCitations, setActiveCitations] = useState([])

  // PDF Viewer Modal State
  const [isPdfModalOpen, setIsPdfModalOpen] = useState(false)
  const [pdfFile, setPdfFile] = useState('')
  const [pdfPage, setPdfPage] = useState(1)

  const handleOpenCitations = (citations) => {
    setActiveCitations(citations || [])
    setIsCitationsOpen(true)
  }

  const handleOpenPdf = (fileName, pageNumber = 1) => {
    setPdfFile(fileName)
    setPdfPage(pageNumber)
    setIsPdfModalOpen(true)
  }

  const handleNewChat = () => {
    setActiveConvId(null)
    setView('chat')
  }

  return (
    <div className="app-container">
      <div className="bg-grid-overlay" />

      {/* Top Navbar */}
      <Navbar
        currentView={currentView}
        setView={setView}
        onNewChat={handleNewChat}
      />

      {/* Main Views */}
      {currentView === 'landing' || currentView === 'rulebooks' ? (
        <LandingPage
          onStartChat={() => setView('chat')}
          onOpenPdf={handleOpenPdf}
        />
      ) : (
        <ChatView
          activeConvId={activeConvId}
          setActiveConvId={setActiveConvId}
          onOpenCitations={handleOpenCitations}
        />
      )}

      {/* Sliding Citations Drawer on the Right */}
      <CitationsSidebar
        isOpen={isCitationsOpen}
        onClose={() => setIsCitationsOpen(false)}
        citations={activeCitations}
        onOpenPdf={handleOpenPdf}
      />

      {/* PDF Direct Page Viewer Modal */}
      <PdfViewerModal
        isOpen={isPdfModalOpen}
        onClose={() => setIsPdfModalOpen(false)}
        pdfFile={pdfFile}
        pageNumber={pdfPage}
      />
    </div>
  )
}
