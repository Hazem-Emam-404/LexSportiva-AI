const API_BASE = (import.meta.env.VITE_API_BASE_URL || '/api').replace(/\/+$/, '')

function getUserId() {
  let uid = localStorage.getItem('lexsportiva_user_id')
  if (!uid) {
    uid = 'usr_' + Math.random().toString(36).substring(2, 10) + Date.now().toString(36)
    localStorage.setItem('lexsportiva_user_id', uid)
  }
  return uid
}

function getHeaders(extraHeaders = {}) {
  return {
    'Content-Type': 'application/json',
    'X-User-ID': getUserId(),
    ...extraHeaders,
  }
}

function syncUserId(res) {
  const headerUid = res.headers?.get('x-user-id') || res.headers?.get('X-User-ID')
  if (headerUid) {
    localStorage.setItem('lexsportiva_user_id', headerUid)
  }
}

export async function fetchConversations() {
  const res = await fetch(`${API_BASE}/conversations`, {
    headers: getHeaders(),
    credentials: 'include',
  })
  syncUserId(res)
  if (!res.ok) throw new Error('Failed to load conversations')
  return res.json()
}

export async function createConversation(title = 'New Chat') {
  const res = await fetch(`${API_BASE}/conversations`, {
    method: 'POST',
    headers: getHeaders(),
    credentials: 'include',
    body: JSON.stringify({ title }),
  })
  syncUserId(res)
  if (!res.ok) throw new Error('Failed to create conversation')
  return res.json()
}

export async function getConversation(conversationId) {
  const res = await fetch(`${API_BASE}/conversations/${conversationId}`, {
    headers: getHeaders(),
    credentials: 'include',
  })
  syncUserId(res)
  if (!res.ok) throw new Error('Failed to load conversation')
  return res.json()
}

export async function deleteConversation(conversationId) {
  const res = await fetch(`${API_BASE}/conversations/${conversationId}`, {
    method: 'DELETE',
    headers: getHeaders(),
    credentials: 'include',
  })
  syncUserId(res)
  if (!res.ok) throw new Error('Failed to delete conversation')
  return res.json()
}

export async function sendMessage(message, conversationId = null) {
  const res = await fetch(`${API_BASE}/chat`, {
    method: 'POST',
    headers: getHeaders(),
    credentials: 'include',
    body: JSON.stringify({
      message,
      conversation_id: conversationId,
    }),
  })
  syncUserId(res)
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
