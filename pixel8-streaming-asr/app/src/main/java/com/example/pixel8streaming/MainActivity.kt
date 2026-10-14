package com.example.pixel8streaming

import android.Manifest
import android.app.Activity
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Bundle
import android.os.StatFs
import android.os.SystemClock
import android.util.Log
import android.view.WindowManager
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.ScrollView
import android.widget.Spinner
import android.widget.TextView
import com.k2fsa.sherpa.onnx.EndpointConfig
import com.k2fsa.sherpa.onnx.EndpointRule
import com.k2fsa.sherpa.onnx.FeatureConfig
import com.k2fsa.sherpa.onnx.OnlineModelConfig
import com.k2fsa.sherpa.onnx.OnlineRecognizer
import com.k2fsa.sherpa.onnx.OnlineRecognizerConfig
import com.k2fsa.sherpa.onnx.OnlineStream
import com.k2fsa.sherpa.onnx.OnlineTransducerModelConfig
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream
import org.apache.commons.compress.compressors.bzip2.BZip2CompressorInputStream
import java.io.BufferedInputStream
import java.io.BufferedOutputStream
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest
import java.util.Locale
import java.util.concurrent.ArrayBlockingQueue
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import kotlin.concurrent.thread
import kotlin.math.max

/**
 * Pixel 8 offline Nemotron 3.5 diagnostic:
 * TRUE streaming (stateful RNN-T), separate AudioRecord capture & ASR threads.
 * No ASR batch windows, no translator, no TTS.
 */
class MainActivity : Activity() {
    private val rate = 16000
    private val packetSamples = 1600 // 100 ms each
    private val queue = ArrayBlockingQueue<ShortArray>(80) // max 8 s of unprocessed audio
    private val droppedPackets = AtomicInteger(0)

    private lateinit var variantSpinner: Spinner
    private lateinit var languageSpinner: Spinner
    private lateinit var downloadButton: Button
    private lateinit var startButton: Button
    private lateinit var stopButton: Button
    private lateinit var statusView: TextView
    private lateinit var liveView: TextView
    private lateinit var historyView: TextView
    private lateinit var metricsView: TextView
    private lateinit var progress: ProgressBar

    @Volatile private var running = false
    @Volatile private var busy = false
    @Volatile private var microphone: AudioRecord? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        buildUi()
        postStatus("Выберите 160 или 560 мс, скачайте модель и запустите streaming.")
    }

    private fun buildUi() {
        val pad = (16 * resources.displayMetrics.density).toInt()
        val box = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(pad, pad, pad, pad)
        }
        val scroll = ScrollView(this).apply { addView(box) }
        setContentView(scroll)

        fun label(value: String, size: Float = 15f): TextView {
            return TextView(this).apply {
                text = value
                textSize = size
                setPadding(0, 7, 0, 7)
                box.addView(this)
            }
        }

        label("Nemotron 3.5 • настоящий streaming", 21f)
        label("Pixel 8 · русский / немецкий · без облака после загрузки")
        label("Размер акустического чанка")
        variantSpinner = Spinner(this).also {
            it.adapter = ArrayAdapter(
                this, android.R.layout.simple_spinner_dropdown_item,
                listOf("160 мс (быстрее)", "560 мс (точнее)")
            )
            box.addView(it)
        }
        label("Язык нового потока")
        languageSpinner = Spinner(this).also {
            it.adapter = ArrayAdapter(
                this, android.R.layout.simple_spinner_dropdown_item,
                listOf("AUTO", "RU", "DE")
            )
            box.addView(it)
        }

        downloadButton = Button(this).apply {
            text = "Скачать модель (~0,5 ГБ)"
            setOnClickListener { downloadSelectedModel() }
            box.addView(this)
        }
        progress = ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal).apply {
            max = 100
            progress = 0
            box.addView(this)
        }
        startButton = Button(this).apply {
            text = "Запустить streaming"
            setOnClickListener { startRecognition() }
            box.addView(this)
        }
        stopButton = Button(this).apply {
            text = "Остановить"
            isEnabled = false
            setOnClickListener { stopRecognition() }
            box.addView(this)
        }
        statusView = label("Статус")
        label("PARTIAL (обновляется прямо во время речи)", 17f)
        liveView = label("—", 19f)
        label("FINAL (после завершения сегмента)", 17f)
        historyView = label("—")
        label("Диагностика", 17f)
        metricsView = label("RTF — | очередь — | потери —")
        label(
            "RTF = время декодирования / длительность обработанного аудио. " +
            "Очередь = примерное отставание необработанного звука. " +
            "AUTO выбирает язык внутри модели; пометка RU/DE на экране — только по алфавиту."
        )
    }

    private fun variant(): String =
        if (variantSpinner.selectedItemPosition == 0) "160" else "560"

    private fun language(): String = when (languageSpinner.selectedItemPosition) {
        1 -> "ru-RU"
        2 -> "de-DE"
        else -> "auto"
    }

    private fun modelDirectory(v: String): File = File(filesDir, "models/" + v)

    private fun installed(v: String): Boolean {
        val dir = modelDirectory(v)
        return File(dir, "encoder.int8.onnx").length() > 600_000_000L &&
            File(dir, "decoder.int8.onnx").length() > 13_000_000L &&
            File(dir, "joiner.int8.onnx").length() > 8_000_000L &&
            File(dir, "tokens.txt").length() > 100_000L
    }

    private fun setUiState(active: Boolean, downloading: Boolean) {
        startButton.isEnabled = !active && !downloading
        stopButton.isEnabled = active
        downloadButton.isEnabled = !active && !downloading
        variantSpinner.isEnabled = !active && !downloading
        languageSpinner.isEnabled = !active && !downloading
    }

    private fun postStatus(message: String) {
        Log.i("Pixel8Nemotron", message)
        runOnUiThread { statusView.text = message }
    }

    private fun downloadSelectedModel() {
        if (busy || running) return
        val v = variant()
        if (installed(v)) {
            postStatus("Модель " + v + " мс уже установлена. Можно запускать без интернета.")
            return
        }
        busy = true
        setUiState(active = false, downloading = true)
        thread(name = "nemotron-model-installer") {
            try {
                val available = StatFs(filesDir.absolutePath).availableBytes
                require(available >= 1_400_000_000L) {
                    "Недостаточно места: нужно минимум 1,4 ГБ свободно."
                }
                installModel(v)
                postStatus("Модель " + v + " мс установлена. Теперь распознавание офлайн.")
            } catch (t: Throwable) {
                Log.e("Pixel8Nemotron", "Model download/extract failed", t)
                postStatus("Ошибка установки: " + t.javaClass.simpleName + ": " + t.message)
            } finally {
                busy = false
                runOnUiThread { setUiState(active = false, downloading = false) }
            }
        }
    }

    private fun installModel(v: String) {
        val base = File(filesDir, "models").apply { mkdirs() }
        val partial = File(base, v + ".staging")
        val archive = File(base, v + ".tar.bz2")
        partial.deleteRecursively()
        check(partial.mkdirs()) { "Cannot create staging directory" }
        val finalDir = modelDirectory(v)
        val modelName = "sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-" +
            v + "ms-int8-2026-06-11.tar.bz2"
        val modelUrl = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/" +
            modelName

        try {
            postStatus("Загрузка модели " + v + " мс...")
            val connection = URL(modelUrl).openConnection() as HttpURLConnection
            connection.connectTimeout = 30000
            connection.readTimeout = 90000
            connection.instanceFollowRedirects = true
            connection.setRequestProperty("Accept-Encoding", "identity")
            try {
                check(connection.responseCode in 200..299) {
                    "HTTP " + connection.responseCode
                }
                val total = connection.contentLengthLong
                val sha256 = MessageDigest.getInstance("SHA-256")
                var count = 0L
                var lastUpdate = 0L
                connection.inputStream.use { input ->
                    FileOutputStream(archive).use { output ->
                        val buffer = ByteArray(128 * 1024)
                        while (true) {
                            val n = input.read(buffer)
                            if (n < 0) break
                            output.write(buffer, 0, n)
                            sha256.update(buffer, 0, n)
                            count += n
                            val now = SystemClock.elapsedRealtime()
                            if (now - lastUpdate >= 500L) {
                                lastUpdate = now
                                val percent = if (total > 0) (count * 100 / total).toInt()
                                    .coerceIn(0, 100) else 0
                                runOnUiThread {
                                    progress.progress = percent
                                    statusView.text = "Загружено " +
                                        (count / 1_000_000L) + " МБ" +
                                        (if (total > 0) " / " + (total / 1_000_000L) + " МБ" else "")
                                }
                            }
                        }
                    }
                }
                require(count > 400_000_000L) { "Архив модели слишком маленький" }
                if (v == "560") {
                    require(count == 475271763L) { "Неверный размер архива 560 мс" }
                    val hex = sha256.digest().joinToString("") {
                        "%02x".format(Locale.US, it.toInt() and 255)
                    }
                    require(
                        hex == "c6bf5e0df765f9d5b43bc9e0536d4b4b3e7d40bdf5ecf13e45f134c51c05ae3a"
                    ) { "Не совпал SHA-256 архива" }
                }
            } finally {
                connection.disconnect()
            }

            postStatus("Распаковка (сохраняются только 4 необходимых файла)...")
            val allowed = setOf(
                "encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"
            )
            val extracted = mutableSetOf<String>()
            FileInputStream(archive).use { fileInput ->
                BZip2CompressorInputStream(BufferedInputStream(fileInput)).use { bz ->
                    TarArchiveInputStream(bz).use { tar ->
                        var entry = tar.nextTarEntry
                        while (entry != null) {
                            // Whitelist basename and ignore test samples, README, links, etc.
                            val name = entry.name.substringAfterLast('/')
                            if (entry.isFile && name in allowed) {
                                require(extracted.add(name)) { "Duplicate model file" }
                                BufferedOutputStream(FileOutputStream(File(partial, name))).use { out ->
                                    tar.copyTo(out, 128 * 1024)
                                }
                            }
                            entry = tar.nextTarEntry
                        }
                    }
                }
            }

            require(extracted == allowed) {
                "Missing model files: " + (allowed - extracted).joinToString(", ")
            }
            require(File(partial, "encoder.int8.onnx").length() > 600_000_000L)
            require(File(partial, "decoder.int8.onnx").length() > 13_000_000L)
            require(File(partial, "joiner.int8.onnx").length() > 8_000_000L)
            require(File(partial, "tokens.txt").length() > 100_000L)
            finalDir.deleteRecursively()
            check(partial.renameTo(finalDir)) { "Failed to finalize model installation" }
            runOnUiThread { progress.progress = 100 }
        } finally {
            archive.delete()
            partial.deleteRecursively()
        }
    }

    private fun startRecognition() {
        if (busy || running) return
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) !=
            PackageManager.PERMISSION_GRANTED
        ) {
            requestPermissions(arrayOf(Manifest.permission.RECORD_AUDIO), 10)
            postStatus("Разрешите доступ к микрофону и нажмите запуск ещё раз.")
            return
        }
        val v = variant()
        if (!installed(v)) {
            postStatus("Сначала скачайте модель " + v + " мс.")
            return
        }
        val lang = language()
        queue.clear()
        droppedPackets.set(0)
        running = true
        setUiState(active = true, downloading = false)
        liveView.text = "—"
        historyView.text = "—"
        metricsView.text = "Загрузка ONNX..."
        thread(name = "nemotron-asr") { runRecognizer(v, lang) }
    }

    private fun stopRecognition() {
        running = false
        postStatus("Остановка: завершаю поток и обрабатываю остаток очереди...")
        // Interrupt blocking read while inference worker is decoding.
        try { microphone?.stop() } catch (_: Throwable) { }
    }

    private fun scriptLabel(text: String): String {
        return when {
            text.any { it in 'А'..'я' || it == 'ё' || it == 'Ё' } -> "RU? (кириллица)"
            text.any { it.isLetter() } -> "DE?/Latin (эвристика)"
            else -> "не определён"
        }
    }

    private fun runRecognizer(v: String, lang: String) {
        var recognizer: OnlineRecognizer? = null
        var stream: OnlineStream? = null
        var recorder: AudioRecord? = null
        var captureThread: Thread? = null
        var audioMs = 0.0
        var decodeMs = 0.0
        var firstPartialMs = -1L
        var sessionStart = 0L
        var latest = ""
        var lastUi = 0L
        var segments = 0
        var endpointEvents = 0
        var maxSeenTokens = 0
        var peakDb = -100.0
        var speechLikePackets = 0
        val history = StringBuilder()

        try {
            postStatus("Инициализация Nemotron " + v + " мс / " + lang + "...")
            val dir = modelDirectory(v)
            val config = OnlineRecognizerConfig(
                featConfig = FeatureConfig(sampleRate = rate, featureDim = 80),
                modelConfig = OnlineModelConfig(
                    transducer = OnlineTransducerModelConfig(
                        encoder = File(dir, "encoder.int8.onnx").absolutePath,
                        decoder = File(dir, "decoder.int8.onnx").absolutePath,
                        joiner = File(dir, "joiner.int8.onnx").absolutePath
                    ),
                    tokens = File(dir, "tokens.txt").absolutePath,
                    numThreads = 2,
                    provider = "cpu"
                ),
                endpointConfig = EndpointConfig(
                    rule1 = EndpointRule(false, 2.4f, 0.0f),
                    rule2 = EndpointRule(true, 0.9f, 0.0f),
                    rule3 = EndpointRule(false, 0f, 20f)
                ),
                enableEndpoint = true,
                decodingMethod = "greedy_search"
            )
            recognizer = OnlineRecognizer(config = config)
            if (!running) return
            stream = recognizer.createStream()
            stream.setOption("language", lang)
            // Verify that the native stream accepted the string, not that the model mapped it.
            val actualPrompt = stream.getOption("language")
            Log.i("Pixel8Nemotron", "Requested locale=$lang, stream option=$actualPrompt")
            require(actualPrompt == lang) { "Language prompt was not stored by native stream" }

            val minSize = AudioRecord.getMinBufferSize(
                rate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT
            )
            require(minSize > 0) { "Unsupported microphone sample rate" }
            recorder = AudioRecord(
                MediaRecorder.AudioSource.MIC, rate, AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT, max(minSize, packetSamples * 2 * 8)
            )
            require(recorder.state == AudioRecord.STATE_INITIALIZED) {
                "AudioRecord could not initialize"
            }
            microphone = recorder
            if (!running) return
            recorder.startRecording()
            sessionStart = SystemClock.elapsedRealtime()
            postStatus("Слушаю. ASR обрабатывает PCM во время речи. " +
                "Режим " + lang + ", " + v + " мс.")

            val captureRecorder = recorder
            captureThread = thread(name = "nemotron-audio-capture") {
                val buffer = ShortArray(packetSamples)
                try {
                    while (running) {
                        val n = captureRecorder.read(
                            buffer, 0, buffer.size, AudioRecord.READ_BLOCKING
                        )
                        if (n > 0) {
                            val item = buffer.copyOf(n)
                            if (!queue.offer(item)) {
                                queue.poll() // oldest packet dropped; mic must not block
                                droppedPackets.incrementAndGet()
                                queue.offer(item)
                            }
                        } else if (n < 0 && running) {
                            Log.e("Pixel8Nemotron", "AudioRecord.read error=" + n)
                            running = false
                        }
                    }
                } catch (t: Throwable) {
                    Log.e("Pixel8Nemotron", "Capture failed", t)
                    running = false
                }
            }

            // TRUE streaming: keep ONE stateful OnlineStream and feed only NEW PCM.
            while (running || queue.isNotEmpty()) {
                val pcm = queue.poll(250, TimeUnit.MILLISECONDS) ?: continue
                val samples = FloatArray(pcm.size) { pcm[it] / 32768.0f }
                val t0 = SystemClock.elapsedRealtimeNanos()
                stream.acceptWaveform(samples, rate)
                while (recognizer.isReady(stream)) {
                    recognizer.decode(stream)
                }
                decodeMs += (SystemClock.elapsedRealtimeNanos() - t0) / 1_000_000.0
                audioMs += samples.size * 1000.0 / rate

                val result = recognizer.getResult(stream)
                val text = result.text.trim()
                maxSeenTokens = max(maxSeenTokens, result.tokens.size)
                var sumOfSquares = 0.0
                for (sample in samples) {
                    sumOfSquares += sample.toDouble() * sample.toDouble()
                }
                val db = 20.0 * kotlin.math.log10(
                    kotlin.math.sqrt(sumOfSquares / samples.size) + 1e-7
                )
                peakDb = kotlin.math.max(peakDb, db)
                if (db > -42.0) speechLikePackets++
                if (text.isNotEmpty() && text != latest) {
                    latest = text
                    if (firstPartialMs < 0) {
                        firstPartialMs = SystemClock.elapsedRealtime() - sessionStart
                    }
                }

                val endpoint = recognizer.isEndpoint(stream)
                if (endpoint) {
                    endpointEvents++
                    Log.d("Pixel8Nemotron", "Endpoint=$endpointEvents text.length=${text.length} " +
                        "tokenCount=${result.tokens.size} locale=$lang")
                    if (text.isNotBlank()) {
                        segments++
                        if (history.isNotEmpty()) history.append("\n")
                        history.append(segments).append(". ").append(text)
                    }
                    recognizer.reset(stream)
                    latest = ""
                }

                val now = SystemClock.elapsedRealtime()
                if (endpoint || now - lastUi >= 250) {
                    lastUi = now
                    val rtf = if (audioMs > 0) decodeMs / audioMs else 0.0
                    val metrics = String.format(
                        Locale.US,
                        "RTF декодера: %.2f | обработано: %.1f с\n" +
                        "Очередь: %.1f с (%d пакетов)\n" +
                        "Потеряно: %d пакетов | FINAL: %d\n" +
                        "Первый текст после старта: %s\n" +
                        "Пик микрофона: %.0f dBFS | пакетов с сигналом: %d\n" +
                        "Токенов (макс.): %d | Endpoints: %d | locale=%s",
                        rtf, audioMs / 1000.0,
                        queue.size / 10.0, queue.size,
                        droppedPackets.get(), segments,
                        if (firstPartialMs >= 0) firstPartialMs.toString() + " мс" else "—",
                        peakDb, speechLikePackets,
                        maxSeenTokens, endpointEvents, lang
                    )
                    val partial = if (latest.isBlank()) "(слушаю)" else
                        latest + "\n[" + scriptLabel(latest) + "]"
                    val finalHistory = history.toString().ifBlank { "—" }
                    runOnUiThread {
                        liveView.text = partial
                        historyView.text = finalHistory
                        metricsView.text = metrics
                    }
                    Log.d("Pixel8NemotronMetrics", metrics.replace('\n', ' '))
                }
            }

            // Flush tail of the live stream without rerunning the entire recording.
            stream.inputFinished()
            val tailStart = SystemClock.elapsedRealtimeNanos()
            while (recognizer.isReady(stream)) {
                recognizer.decode(stream)
            }
            decodeMs += (SystemClock.elapsedRealtimeNanos() - tailStart) / 1_000_000.0
            val tail = recognizer.getResult(stream).text.trim()
            if (tail.isNotBlank()) {
                if (history.isNotEmpty()) history.append("\n")
                history.append(segments + 1).append(". ").append(tail)
            }
            val finalRtf = if (audioMs > 0) decodeMs / audioMs else 0.0
            val finalHistory = history.toString().ifBlank { "—" }
            runOnUiThread {
                historyView.text = finalHistory
                metricsView.text = String.format(
                    Locale.US,
                    "FINAL RTF: %.2f | аудио: %.1f с | потерь: %d\n" +
                    "Пик микрофона: %.0f dBFS | пакетов с сигналом: %d\n" +
                    "Токенов (макс.): %d | Endpoints: %d | язык: %s",
                    finalRtf, audioMs / 1000.0, droppedPackets.get(),
                    peakDb, speechLikePackets, maxSeenTokens, endpointEvents, lang
                )
            }
            postStatus(
                if (history.isEmpty() && speechLikePackets >= 8) {
                    "Текст пустой, хотя микрофон зафиксировал сигнал. " +
                    "Проверьте max tokens/endpoints в диагностике."
                } else "Остановлено. Результат получен локально, без API."
            )
        } catch (t: Throwable) {
            Log.e("Pixel8Nemotron", "Streaming ASR failed", t)
            postStatus("Ошибка ASR: " + t.javaClass.simpleName + ": " + t.message)
        } finally {
            running = false
            try { recorder?.stop() } catch (_: Throwable) { }
            try { captureThread?.join(1000) } catch (_: Throwable) { }
            try { recorder?.release() } catch (_: Throwable) { }
            microphone = null
            try { stream?.release() } catch (_: Throwable) { }
            try { recognizer?.release() } catch (_: Throwable) { }
            queue.clear()
            runOnUiThread { setUiState(active = false, downloading = false) }
        }
    }

    override fun onDestroy() {
        running = false
        try { microphone?.stop() } catch (_: Throwable) { }
        super.onDestroy()
    }
}
