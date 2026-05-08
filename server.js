import express from 'express'
import makeWASocket, {
  useMultiFileAuthState,
  fetchLatestBaileysVersion,
  DisconnectReason,
  downloadContentFromMessage
} from '@whiskeysockets/baileys'
import qrcode  from 'qrcode-terminal'
import axios   from 'axios'
import pino    from 'pino'
import fs      from 'fs'
import path    from 'path'

const app = express()
app.use(express.json())

const CONFIG = {
  PYTHON_URL:      'http://127.0.0.1:8000/agent',
  POLL_URL:        'http://127.0.0.1:8000/poll',   // ← nouveau endpoint poll
  TIMEOUT:         12000,    // 12s — juste pour recevoir l ACK (pas le résultat final)
  POLL_INTERVAL:   3000,     // poll toutes les 3s
  POLL_MAX_TRIES:  60,       // 60 × 3s = 3 minutes max
  RECONNECT_DELAY: 5000,
  MAX_RECONNECT:   10
}

const AUTHORIZED = [
  '217252381089895',
  '22222306004',
  '280444201349330',
]

function isAuthorized(jid) {
  const jidClean = jid.replace('@s.whatsapp.net','').replace('@lid','').replace('@c.us','')
  return AUTHORIZED.some(num => jidClean === num || jid.startsWith(num))
}

let sock
let reconnectCount = 0

// ── Polling actif par userId ──────────────────────────────────────────────────
// Map userId → { intervalId, tries }
const activePolls = new Map()

function startPolling(from) {
  // Évite de démarrer deux polls pour le même utilisateur
  if (activePolls.has(from)) return

  let tries = 0
  console.log(`🔄 Poll démarré pour [${from}]`)

  const intervalId = setInterval(async () => {
    tries++

    // Timeout total dépassé
    if (tries > CONFIG.POLL_MAX_TRIES) {
      console.warn(`⏰ [${from}] Poll timeout`)
      clearInterval(intervalId)
      activePolls.delete(from)
      try {
        await sock.sendMessage(from, { text: '⚠️ Traitement trop long, réessayez.' })
      } catch (_) {}
      return
    }

    try {
      const res   = await axios.get(
        `${CONFIG.POLL_URL}/${encodeURIComponent(from)}`,
        { timeout: 35000 }
      )
      const ready = res.data?.ready
      const reply = res.data?.reply
      const pdfPath  = res.data?.pdf_path

      // Pas encore prêt
      if (!reply && !pdfPath) return

      // Réponse prête → stop poll
      clearInterval(intervalId)
      activePolls.delete(from)
      console.log(`✅ [${from}] Réponse reçue après ${tries} poll(s)`)

      // ── CAS PDF ────────────────────────────────────────────────────────────
      if (pdfPath) {
        const pdfBuffer = fs.readFileSync(pdfPath)
        const pdfName   = path.basename(pdfPath)
        await sock.sendMessage(from, { text: reply || '📄 PDF prêt' })
        await sock.sendMessage(from, {
          document: pdfBuffer,
          mimetype: 'application/pdf',
          fileName: pdfName,
        })
        console.log(`📄 PDF envoyé: ${pdfName}`)
        try { fs.unlinkSync(pdfPath) } catch (_) {}
        return
      }

      // ── CAS TEXTE ──────────────────────────────────────────────────────────
      if(reply) await sock.sendMessage(from, { text: reply })
      if(reply) console.log(`📤 BOT: ${reply.substring(0, 120)}`)

    } catch (err) {
      // Erreur réseau ponctuelle → on réessaie au prochain cycle
      console.warn(`⚠️  Poll [${from}] erreur: ${err.message}`)
    }
  }, CONFIG.POLL_INTERVAL)

  activePolls.set(from, { intervalId, tries: 0 })
}

// ── Bot WhatsApp ──────────────────────────────────────────────────────────────

async function startBot() {
  const { state, saveCreds } = await useMultiFileAuthState('./auth_session')
  const { version }          = await fetchLatestBaileysVersion()

  sock = makeWASocket({
    version,
    auth:   state,
    logger: pino({ level: 'silent' }),
    browser: ['Ubuntu', 'Chrome', '1.0'],
    syncFullHistory:                false,
    markOnlineOnConnect:            false,
    generateHighQualityLinkPreview: false,
    defaultQueryTimeoutMs:          60000
  })

  sock.ev.on('creds.update', saveCreds)

  sock.ev.on('connection.update', ({ connection, qr, lastDisconnect }) => {
    if (qr) {
      console.log('\n📱 SCANNEZ CE QR CODE AVEC WHATSAPP:\n')
      qrcode.generate(qr, { small: true })
    }
    if (connection === 'open') {
      console.log('✅ WhatsApp CONNECTE')
      reconnectCount = 0
    }
    if (connection === 'close') {
      const code = lastDisconnect?.error?.output?.statusCode
      console.log('❌ Connexion fermee:', code)
      if (code === DisconnectReason.loggedOut) {
        console.log('Deconnecte. Supprimez auth_session/ et relancez.')
        return
      }
      reconnectCount++
      if (reconnectCount > CONFIG.MAX_RECONNECT) return
      console.log(`🔁 Reconnexion ${reconnectCount}/${CONFIG.MAX_RECONNECT}...`)
      setTimeout(startBot, CONFIG.RECONNECT_DELAY)
    }
  })

  sock.ev.on('messages.upsert', async ({ messages }) => {
    const msg = messages?.[0]
    if (!msg?.message) return
    if (msg.key.fromMe)  return

    const from = msg.key.remoteJid

    if (!isAuthorized(from)) {
      console.log(`🚫 Ignore: ${from}`)
      return
    }

    const text = (
      msg.message.conversation              ||
      msg.message.extendedTextMessage?.text ||
      msg.message.imageMessage?.caption     ||
      ''
    ).trim()

    let imageBase64 = null
    if (msg.message.imageMessage) {
      try {
        const stream = await downloadContentFromMessage(
          msg.message.imageMessage, 'image'
        )
        let buffer = Buffer.from([])
        for await (const chunk of stream) buffer = Buffer.concat([buffer, chunk])
        imageBase64 = buffer.toString('base64')
        console.log('🖼️ Image recue:', buffer.length, 'bytes')
      } catch (err) {
        console.log('⚠️ Erreur image:', err.message)
      }
    }

    if (!text && !imageBase64) return
    console.log(`\n📩 [${from}]: ${text || '(image)'}`)

    try {
      const { data } = await axios.post(
        CONFIG.PYTHON_URL,
        { message: text, user: from, image: imageBase64 },
        { timeout: CONFIG.TIMEOUT }
      )

      // ── Réponse immédiate avec résultat complet (mode texte ou 1 image rapide)
      if (data.reply) {

        // CAS PDF
        if (data.pdf_path) {
          const pdfBuffer = fs.readFileSync(data.pdf_path)
          const pdfName   = path.basename(data.pdf_path)
          await sock.sendMessage(from, { text: data.reply })
          await sock.sendMessage(from, {
            document: pdfBuffer,
            mimetype: 'application/pdf',
            fileName: pdfName,
          })
          console.log(`📄 PDF envoye: ${pdfName}`)
          try { fs.unlinkSync(data.pdf_path) } catch (_) {}
          return
        }

        // CAS TEXTE NORMAL
        await sock.sendMessage(from, { text: data.reply })
        console.log(`📤 BOT: ${data.reply.substring(0, 120)}`)
        return
      }

      // ── Traitement async en cours (images groupées) → ACK + polling ─────────
      if (data.status === 'processing' || data.status === 'queued') {

        // Envoyer l ACK uniquement pour le 1er webhook du groupe
        if (data.status === 'processing' && data.ack) {
          await sock.sendMessage(from, { text: data.ack })
          console.log(`📤 ACK envoyé → [${from}]`)
        }

        // Démarrer le polling pour récupérer la réponse finale
        startPolling(from)
        return
      }

    } catch (err) {
      console.log('❌ ERREUR AGENT:', err.message)
      // Ne pas envoyer d erreur si c est juste un timeout réseau ponctuel
      if (!err.message.includes('timeout')) {
        await sock.sendMessage(from, { text: '⚠️ Service indisponible, reessayez.' })
      }
    }
  })
}

startBot()

app.get('/health', (req, res) => res.json({ status: 'ok' }))
app.listen(3000, () => console.log('🚀 Server running on :3000'))