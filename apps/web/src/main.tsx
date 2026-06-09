import React from 'react'
import ReactDOM from 'react-dom/client'
import './styles.css'

type GalleryItem = { id: string; name: string; path: string; url: string; mime: string }

type ApiResponse = {
  answer?: string
  final_answer?: string
  raw_output?: string
  label_space?: string
  task_type?: string
  prompt?: string
  text?: string
  result?: string
  response?: string
  detail?: string
}

type HistoryItem = {
  question: string
  answer: string
}

type TaskType = 'yes_no' | 'count' | 'location'
type LabelSpace = 'yes_no' | 'count_4' | 'location_3'

const promptHints = [
  'Can you see a worker wearing a safety helmet?',
  'How many workers wear safety helmets?',
  'Is any worker wearing a safety vest?',
  'How many workers are in the image?',
  'Where is the helmet-wearing worker?',
  'Where is the vest-wearing worker?',
]

const presetPrompts = [
  'Can you see a worker wearing a safety helmet?',
  'Is any worker wearing a safety vest?',
  'How many workers are in the image?',
  'Where is the helmet-wearing worker?',
]

const splitWords = (value: string) => value.trim().split(/\s+/).filter(Boolean)

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL?.replace(/\/$/, '') || ''

const apiUrl = (path: string) => {
  const normalizedPath = path.startsWith('/') ? path : `/${path}`
  return `${API_BASE_URL}${normalizedPath}`
}

const resolveImageUrl = (url: string) => {
  if (/^https?:\/\//i.test(url) || url.startsWith('data:') || url.startsWith('blob:')) {
    return url
  }
  return apiUrl(url)
}

const toDataUrl = async (url: string) => {
  if (url.startsWith('data:')) return url
  const response = await fetch(resolveImageUrl(url))
  if (!response.ok) throw new Error(`HTTP ${response.status}`)
  const blob = await response.blob()
  return await new Promise<string>((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(typeof reader.result === 'string' ? reader.result : '')
    reader.onerror = () => reject(new Error('Failed to read image blob'))
    reader.readAsDataURL(blob)
  })
}

const joinWords = (words: string[]) => words.join(' ')

const randomPromptFromBank = () => promptHints[Math.floor(Math.random() * promptHints.length)]

const getPromptCompletion = (value: string, fallbackPrompt: string) => {
  if (!value.trim()) {
    return splitWords(fallbackPrompt)[0] || ''
  }

  const normalizedValue = value.toLowerCase()
  const hasTrailingSpace = /\s$/.test(value)
  const inputWords = splitWords(normalizedValue)
  if (inputWords.length === 0) return ''

  for (const promptHint of promptHints) {
    const candidateWords = splitWords(promptHint)
    const normalizedCandidateWords = splitWords(promptHint.toLowerCase())
    if (hasTrailingSpace) {
      if (inputWords.length >= normalizedCandidateWords.length) continue
      const matches = inputWords.every((word, index) => normalizedCandidateWords[index] === word)
      if (!matches) continue
      return candidateWords[inputWords.length] || ''
    }

    const prefixWords = inputWords.slice(0, -1)
    const partialWord = inputWords[inputWords.length - 1]
    if (prefixWords.length >= normalizedCandidateWords.length) continue
    const prefixMatches = prefixWords.every((word, index) => normalizedCandidateWords[index] === word)
    if (!prefixMatches) continue
    const candidateWord = candidateWords[prefixWords.length]
    const normalizedCandidateWord = normalizedCandidateWords[prefixWords.length]
    if (!normalizedCandidateWord.startsWith(partialWord)) continue

    return candidateWord.slice(partialWord.length)
  }

  return ''
}

function App() {
  const [galleryImages, setGalleryImages] = React.useState<GalleryItem[]>([])
  const [selectedImage, setSelectedImage] = React.useState<string>('')
  const [prompt, setPrompt] = React.useState('')
  const [predictedTaskType, setPredictedTaskType] = React.useState<TaskType>('yes_no')
  const [predictedLabelSpace, setPredictedLabelSpace] = React.useState<LabelSpace>('yes_no')
  const [manualTaskType, setManualTaskType] = React.useState<TaskType | null>(null)
  const [manualLabelSpace, setManualLabelSpace] = React.useState<LabelSpace | null>(null)
  const [result, setResult] = React.useState('')
  const [debugInfo, setDebugInfo] = React.useState('')
  const [loading, setLoading] = React.useState(false)
  const [history, setHistory] = React.useState<HistoryItem[]>([])
  const [videoStream, setVideoStream] = React.useState<MediaStream | null>(null)
  const [error, setError] = React.useState('')
  const [presetIndex, setPresetIndex] = React.useState(0)
  const [presetFade, setPresetFade] = React.useState(false)
  const [imageSource, setImageSource] = React.useState<'gallery' | 'upload' | 'camera'>('gallery')
  const [uploadedPreview, setUploadedPreview] = React.useState<string | null>(null)
  const [cameraPreview, setCameraPreview] = React.useState<string | null>(null)
  const fileInputRef = React.useRef<HTMLInputElement | null>(null)
  const videoRef = React.useRef<HTMLVideoElement | null>(null)
  const canvasRef = React.useRef<HTMLCanvasElement | null>(null)

  const currentImage = cameraPreview || uploadedPreview || selectedImage
  const promptBackground = presetPrompts[presetIndex]

  const inlineCompletion = React.useMemo(() => {
    return getPromptCompletion(prompt, promptBackground)
  }, [prompt, promptBackground])

  const taskType = manualTaskType || predictedTaskType
  const labelSpace = manualLabelSpace || predictedLabelSpace

  React.useEffect(() => {
    fetch(apiUrl('/api/gallery'))
      .then((res) => res.json())
      .then((data) => {
        if (Array.isArray(data.items)) {
          const nextItems = data.items.map((item: GalleryItem) => ({
            ...item,
            url: resolveImageUrl(item.url),
          }))
          setGalleryImages(nextItems)
          if (nextItems[0]?.url) setSelectedImage(nextItems[0].url)
        }
      })
      .catch(() => setError('无法加载图库，请检查后端 /api/gallery。'))
  }, [])

  const stopCamera = React.useCallback(() => {
    setVideoStream((stream) => {
      stream?.getTracks().forEach((track) => track.stop())
      return null
    })
    if (videoRef.current) videoRef.current.srcObject = null
  }, [])

  React.useEffect(() => {
    if (imageSource !== 'camera') stopCamera()
  }, [imageSource, stopCamera])

  React.useEffect(() => {
    const video = videoRef.current
    if (!video || imageSource !== 'camera' || !videoStream) return

    if (video.srcObject !== videoStream) {
      video.srcObject = videoStream
    }

    void video.play().catch(() => {})

    return () => {
      if (video.srcObject === videoStream) {
        video.pause()
        video.srcObject = null
      }
    }
  }, [imageSource, videoStream, cameraPreview, stopCamera])

  React.useEffect(() => {
    return () => stopCamera()
  }, [stopCamera])

  React.useEffect(() => {
    const timer = window.setInterval(() => {
      setPresetFade(true)
      window.setTimeout(() => {
        setPresetIndex((index) => (index + 1) % presetPrompts.length)
        setPresetFade(false)
      }, 180)
    }, 4200)
    return () => window.clearInterval(timer)
  }, [])

  const applySuggestion = (value: string) => {
    setPrompt(value)
  }

  const taskTypeToLabelSpace = (nextTaskType: TaskType): LabelSpace => {
    if (nextTaskType === 'count') return 'count_4'
    if (nextTaskType === 'location') return 'location_3'
    return 'yes_no'
  }

  const applyTaskType = (nextTaskType: TaskType) => {
    setManualTaskType(nextTaskType)
    setManualLabelSpace(taskTypeToLabelSpace(nextTaskType))
  }

  const clearManualOverride = () => {
    setManualTaskType(null)
    setManualLabelSpace(null)
  }

  const handlePromptKeyDown = (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Tab') {
      event.preventDefault()
      if (inlineCompletion) setPrompt((value) => value + inlineCompletion + ' ')
      return
    }
  }

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = () => {
      const preview = typeof reader.result === 'string' ? reader.result : ''
      if (!preview) {
        setError('图片读取失败，请重试。')
        return
      }
      setUploadedPreview(preview)
      setCameraPreview(null)
      setImageSource('upload')
      setError('')
    }
    reader.onerror = () => setError('图片读取失败，请重试。')
    reader.readAsDataURL(file)
  }

  const startCamera = async () => {
    try {
      stopCamera()
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false })
      setVideoStream(stream)
      setImageSource('camera')
      setCameraPreview(null)
      setError('')
    } catch {
      setError('无法开启摄像头，请确认浏览器权限。')
    }
  }

  const captureCurrentCameraFrame = () => {
    const video = videoRef.current
    const canvas = canvasRef.current
    if (!video || !canvas) return ''
    if (video.videoWidth === 0 || video.videoHeight === 0) return ''
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    const ctx = canvas.getContext('2d')
    if (!ctx) return ''
    ctx.drawImage(video, 0, 0)
    const snapshot = canvas.toDataURL('image/png')
    setCameraPreview(snapshot)
    setUploadedPreview(null)
    setError('')
    return snapshot
  }

  const captureCameraFrame = () => {
    const snapshot = captureCurrentCameraFrame()
    if (!snapshot) setError('无法定格当前帧，请稍后重试。')
  }

  const continueShooting = () => {
    if (!videoStream) return
    setImageSource('camera')
    setCameraPreview(null)
    setUploadedPreview(null)
    setError('')
    const video = videoRef.current
    if (video) void video.play().catch(() => {})
  }

  const sendRequest = async () => {
    let imageForRequest = currentImage
    if (imageSource === 'camera') {
      imageForRequest = cameraPreview || captureCurrentCameraFrame()
    }
    if (!imageForRequest) {
      setError('请先选择一张图片。')
      return
    }
    setLoading(true)
    setError('')
    try {
      const imagePayload = await toDataUrl(imageForRequest)
      const payload: Record<string, unknown> = {
        image: imagePayload,
        question: prompt,
      }
      if (manualTaskType) payload.task_type = manualTaskType
      if (manualLabelSpace) payload.label_space = manualLabelSpace
      const response = await fetch(apiUrl('/api/infer'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      const text = await response.text()
      if (!response.ok) throw new Error(text || `HTTP ${response.status}`)
      let data: ApiResponse = {}
      try {
        data = text ? (JSON.parse(text) as ApiResponse) : {}
      } catch {
        data = { answer: text }
      }
      const answer = data.final_answer || data.answer || data.text || data.result || data.response || data.detail || 'No response'
      const display = answer
      setResult(display)
      const nextTaskType = (data.task_type as TaskType | undefined) || predictedTaskType
      const nextLabelSpace = (data.label_space as LabelSpace | undefined) || predictedLabelSpace
      setPredictedTaskType(nextTaskType)
      setPredictedLabelSpace(nextLabelSpace)
      setDebugInfo(`task_type=${nextTaskType}\nlabel_space=${nextLabelSpace}\nprompt=${data.prompt || 'n/a'}\nraw_output=${data.raw_output || 'n/a'}`)
      setHistory((prev) => [{ question: prompt, answer: display }, ...prev].slice(0, 10))
    } catch {
      setError('请求后端失败，请确认后端已启动且 /api/infer 可用。')
    } finally {
      setLoading(false)
    }
  }

  const pickGallery = (url: string) => {
    stopCamera()
    setSelectedImage(url)
    setUploadedPreview(null)
    setCameraPreview(null)
    setImageSource('gallery')
  }

  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">M</div>
          <div>
            <div className="brand-title">Moondream</div>
            <div className="brand-subtitle">图库</div>
          </div>
        </div>

        <div className="sidebar-actions">
          <button className={imageSource === 'gallery' ? 'nav-btn active' : 'nav-btn'} onClick={() => { stopCamera(); setImageSource('gallery'); setUploadedPreview(null); setCameraPreview(null) }}>
            训练/测试图库
          </button>
          <button className={imageSource === 'upload' ? 'nav-btn active' : 'nav-btn'} onClick={() => { stopCamera(); setImageSource('upload'); fileInputRef.current?.click() }}>
            上传图片
          </button>
          <button className={imageSource === 'camera' ? 'nav-btn active' : 'nav-btn'} onClick={startCamera}>
            打开摄像头
          </button>
        </div>

        <div className="sidebar-section">
          <div className="section-title">训练/测试图片</div>
          <div className="gallery-list">
            {galleryImages.map((img) => (
              <button key={img.id} className={selectedImage === img.url ? 'gallery-item active' : 'gallery-item'} onClick={() => pickGallery(img.url)}>
                <img src={img.url} alt={img.name} />
                <div className="gallery-meta">
                  <strong>{img.name}</strong>
                </div>
              </button>
            ))}
          </div>
        </div>
      </aside>

      <main className="workspace">
        <header className="topbar">
          <div>
            <p className="eyebrow">可视化交互界面</p>
            <h1>基于轻量级视觉语言模型的工地安全检测</h1>
            <p className="subtitle">左侧直接读取图库中的图片，右侧输入 prompt 并联调用后端。</p>
          </div>
          <div className="status-card">
            <span className={loading ? 'dot busy' : 'dot'} />
            <div>
              <strong>{loading ? '推理中…' : '就绪'}</strong>
              <p>Vite 代理已指向后端 `:8000`。</p>
            </div>
          </div>
        </header>

        <section className="composer">
          <div className="preview-wrap">
            <div className="preview-frame">
              {imageSource === 'camera' && videoStream && !cameraPreview ? (
                <video ref={videoRef} autoPlay playsInline muted />
              ) : currentImage ? (
                <img src={currentImage} alt="selected" />
              ) : (
                <div className="empty-state">请选择图片</div>
              )}
            </div>
            <input ref={fileInputRef} type="file" accept="image/*" hidden onChange={handleFileChange} />
            {videoStream && (
              <div className="camera-panel">
                <div className="camera-actions">
                  <button onClick={captureCameraFrame}>拍摄</button>
                  <button onClick={continueShooting}>继续拍摄</button>
                </div>
                <canvas ref={canvasRef} hidden />
              </div>
            )}
          </div>

          <div className="prompt-card">
            <div className="prompt-head">
              <div>
                <h2>Prompt</h2>
                <p className="hint">按 Tab 补全，↑ ↓ 切换候选。模型会自动识别三类任务，也可以手动点按钮纠正后再推理。</p>
              </div>
              <div className="task-selector">
                <button className={taskType === 'yes_no' ? 'active' : ''} onClick={() => applyTaskType('yes_no')}>Yes/No</button>
                <button className={taskType === 'count' ? 'active' : ''} onClick={() => applyTaskType('count')}>Count</button>
                <button className={taskType === 'location' ? 'active' : ''} onClick={() => applyTaskType('location')}>Location</button>
              </div>
            </div>

            <div className="prompt-box">
              <div className="prompt-editor-shell">
                <div className="prompt-mirror" aria-hidden="true">
                  <span className={prompt ? 'prompt-mirror-base' : 'prompt-mirror-base prompt-mirror-placeholder'}>{prompt || promptBackground}</span>
                  {prompt && inlineCompletion ? <span className="prompt-mirror-ghost">{inlineCompletion}</span> : null}
                </div>
                <textarea
                  value={prompt}
                  onChange={(e) => {
                    setPrompt(e.target.value)
                  }}
                  onKeyDown={handlePromptKeyDown}
                  spellCheck={false}
                />
              </div>
              <div className={presetFade ? 'preset-strip fade-out' : 'preset-strip fade-in'}>
                <button className="preset-card" onClick={() => applySuggestion(presetPrompts[presetIndex])}>
                  <span className="preset-kicker">你可以这样问</span>
                  <span className="preset-text">{presetPrompts[presetIndex]}</span>
                </button>
              </div>
            </div>

            <div className="actions">
              <button className="primary" onClick={sendRequest} disabled={loading}>发送推理</button>
              <button onClick={() => applySuggestion(randomPromptFromBank())}>示例问题</button>
              <button onClick={clearManualOverride} disabled={!manualTaskType}>恢复自动识别</button>
            </div>

            {error && <p className="error">{error}</p>}
            <div className="result-card">
              <div className="section-title">模型回答</div>
              <div className="result-text">{result || '等待结果...'}</div>
              <div className="result-meta">
                自动分类: {predictedTaskType} / {predictedLabelSpace}
                {manualTaskType ? ` · 人工纠正: ${taskType} / ${labelSpace}` : ''}
              </div>
            </div>
            <div className="debug-card">
              <div className="section-title">Debug</div>
              <pre>{debugInfo || '暂无 debug 信息'}</pre>
            </div>
          </div>
        </section>

        <section className="history-card">
          <div className="section-title">历史对话</div>
          {history.length === 0 ? (
            <p className="muted">暂无记录。</p>
          ) : (
            history.map((item, idx) => (
              <div className="history-item" key={`${item.question}-${idx}`}>
                <strong>{item.question}</strong>
                <span>{item.answer}</span>
              </div>
            ))
          )}
        </section>
      </main>
    </div>
  )
}

ReactDOM.createRoot(document.getElementById('root') as HTMLElement).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
