const API_BASE = (import.meta.env.VITE_API_BASE_URL || '/api').replace(/\/+$/, '')

export async function fetchConversations() {
  const res = await fetch(`${API_BASE}/conversations`, {
    credentials: 'include',
  })
  if (!res.ok) throw new Error('Failed to load conversations')
  return res.json()
}

export async function createConversation(title = 'New Chat') {
  const res = await fetch(`${API_BASE}/conversations`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    credentials: 'include',
    body: JSON.stringify({ title }),
  })
  if (!res.ok) throw new Error('Failed to create conversation')
  return res.json()
}

export async function getConversation(conversationId) {
  const res = await fetch(`${API_BASE}/conversations/${conversationId}`, {
    credentials: 'include',
  })
  if (!res.ok) throw new Error('Failed to load conversation')
  return res.json()
}

export async function deleteConversation(conversationId) {
  const res = await fetch(`${API_BASE}/conversations/${conversationId}`, {
    method: 'DELETE',
    credentials: 'include',
  })
  if (!res.ok) throw new Error('Failed to delete conversation')
  return res.json()
}

export async function sendMessage(message, conversationId = null) {
  const res = await fetch(`${API_BASE}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    credentials: 'include',
    body: JSON.stringify({
      message,
      conversation_id: conversationId,
    }),
  })
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}))
    throw new Error(errorData.detail || 'Failed to send message')
  }
  return res.json()
}

export function getPdfUrl(fileName, page = 1) {
  const cleanName = fileName.replace('.pdf', '') + '.pdf'
  return `${API_BASE}/pdf/${cleanName}#page=${page}`
}
