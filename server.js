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
  TIMEOUT:         30000,
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

      // ─── CAS PDF ───────────────────────────────────────────────────────────
      if (data.pdf_path) {
        const pdfBuffer  = fs.readFileSync(data.pdf_path)
        const pdfName    = path.basename(data.pdf_path)

        // Envoyer message texte d'abord
        await sock.sendMessage(from, { text: data.reply })

        // Envoyer le PDF comme document
        await sock.sendMessage(from, {
          document:  pdfBuffer,
          mimetype:  'application/pdf',
          fileName:  pdfName,
        })
        console.log(`📄 PDF envoye: ${pdfName}`)

        // Nettoyer le fichier temporaire
        fs.unlinkSync(data.pdf_path)
        return
      }

      // ─── CAS TEXTE NORMAL ──────────────────────────────────────────────────
      const reply = data?.reply
      if (!reply) return
      await sock.sendMessage(from, { text: reply })
      console.log(`📤 BOT: ${reply}`)

    } catch (err) {
      console.log('❌ ERREUR AGENT:', err.message)
      await sock.sendMessage(from, { text: '⚠️ Service indisponible, reessayez.' })
    }
  })
}

startBot()

app.get('/health', (req, res) => res.json({ status: 'ok' }))
app.listen(3000, () => console.log('🚀 Server running on :3000'))
