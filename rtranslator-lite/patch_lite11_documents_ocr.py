#!/usr/bin/env python3
from pathlib import Path
import re
import sys

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "RTranslator")
build = ROOT / "app/build.gradle"
manifest = ROOT / "app/src/main/AndroidManifest.xml"
translation_fragment = ROOT / "app/src/main/java/nie/translator/rtranslator/voice_translation/_text_translation/TranslationFragment.java"
translation_layout = ROOT / "app/src/main/res/layout/fragment_translation.xml"
provider_files = [
    manifest,
    ROOT / "app/src/main/java/nie/translator/rtranslator/access/UserDataFragment.java",
    ROOT / "app/src/main/java/nie/translator/rtranslator/settings/SettingsFragment.java",
]

for p in [build, manifest, translation_fragment, translation_layout, *provider_files]:
    if not p.exists():
        raise SystemExit(f"Missing required file: {p}")

java_dir = ROOT / "app/src/main/java/nie/translator/rtranslator/documents"
layout_dir = ROOT / "app/src/main/res/layout"
drawable_dir = ROOT / "app/src/main/res/drawable"
java_dir.mkdir(parents=True, exist_ok=True)
layout_dir.mkdir(parents=True, exist_ok=True)
drawable_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# OCR engine: Tesseract 5, PDF rasterization, bundled core languages and
# official tessdata_fast on-demand fallback for other source languages.
# ---------------------------------------------------------------------------
(java_dir / "OcrEngine.java").write_text(r'''package nie.translator.rtranslator.documents;

import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.pdf.PdfRenderer;
import android.net.Uri;
import android.os.ParcelFileDescriptor;
import android.util.Log;

import com.googlecode.tesseract.android.TessBaseAPI;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;

/** Offline OCR backend for Lite11. Core travel languages are bundled in assets. */
public final class OcrEngine {
    private static final String TAG = "RTranslatorLite11OCR";
    private static final String TESSDATA_COMMIT = "87416418657359cb625c412a48b6e1d6d41c29bd";
    private static final int MAX_RENDER_DIMENSION = 2400;
    private static final int MIN_TRAINEDDATA_BYTES = 100_000;

    public interface ProgressListener {
        void onProgress(int current, int total, String message);
    }

    private OcrEngine() {}

    public static String recognizeBitmap(Context context, Bitmap bitmap, String sourceLanguageCode) throws Exception {
        TessBaseAPI tess = createTess(context, sourceLanguageCode);
        try {
            tess.setImage(bitmap);
            String text = tess.getUTF8Text();
            return clean(text);
        } finally {
            tess.recycle();
        }
    }

    public static String recognizePdf(Context context, Uri uri, String sourceLanguageCode,
                                      ProgressListener progressListener) throws Exception {
        TessBaseAPI tess = createTess(context, sourceLanguageCode);
        StringBuilder out = new StringBuilder();
        try (ParcelFileDescriptor pfd = context.getContentResolver().openFileDescriptor(uri, "r")) {
            if (pfd == null) throw new IOException("Cannot open PDF");
            try (PdfRenderer renderer = new PdfRenderer(pfd)) {
                int total = renderer.getPageCount();
                if (total == 0) return "";
                for (int i = 0; i < total; i++) {
                    if (progressListener != null) {
                        progressListener.onProgress(i + 1, total, "OCR page " + (i + 1) + " / " + total);
                    }
                    try (PdfRenderer.Page page = renderer.openPage(i)) {
                        Bitmap bitmap = renderPage(page);
                        try {
                            tess.setImage(bitmap);
                            String pageText = clean(tess.getUTF8Text());
                            if (!pageText.isEmpty()) {
                                if (out.length() > 0) out.append("\n\n");
                                out.append("--- Page ").append(i + 1).append(" ---\n");
                                out.append(pageText);
                            }
                            tess.clear();
                        } finally {
                            bitmap.recycle();
                        }
                    }
                }
            }
        } finally {
            tess.recycle();
        }
        return out.toString().trim();
    }

    private static Bitmap renderPage(PdfRenderer.Page page) {
        int sourceW = Math.max(1, page.getWidth());
        int sourceH = Math.max(1, page.getHeight());
        float scale = 2.0f;
        int w = Math.round(sourceW * scale);
        int h = Math.round(sourceH * scale);
        int max = Math.max(w, h);
        if (max > MAX_RENDER_DIMENSION) {
            float shrink = MAX_RENDER_DIMENSION / (float) max;
            w = Math.max(1, Math.round(w * shrink));
            h = Math.max(1, Math.round(h * shrink));
        }
        Bitmap bitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(bitmap);
        canvas.drawColor(Color.WHITE);
        page.render(bitmap, null, null, PdfRenderer.Page.RENDER_MODE_FOR_DISPLAY);
        return bitmap;
    }

    private static TessBaseAPI createTess(Context context, String sourceLanguageCode) throws Exception {
        String langSpec = resolveLanguageSpec(sourceLanguageCode);
        for (String code : langSpec.split("\\+")) {
            ensureTrainedData(context, code);
        }
        File root = new File(context.getFilesDir(), "tesseract");
        TessBaseAPI tess = new TessBaseAPI();
        boolean ok = tess.init(root.getAbsolutePath(), langSpec);
        if (!ok) {
            tess.recycle();
            throw new IOException("Tesseract init failed for " + langSpec);
        }
        return tess;
    }

    private static void ensureTrainedData(Context context, String code) throws Exception {
        File root = new File(context.getFilesDir(), "tesseract");
        File tessdata = new File(root, "tessdata");
        if (!tessdata.exists() && !tessdata.mkdirs()) {
            throw new IOException("Cannot create tessdata directory");
        }
        File target = new File(tessdata, code + ".traineddata");
        if (target.isFile() && target.length() >= MIN_TRAINEDDATA_BYTES) return;

        File temporary = new File(tessdata, code + ".traineddata.part");
        if (temporary.exists()) temporary.delete();

        // First try the models bundled by Lite11 CI.
        try (InputStream in = context.getAssets().open("tessdata/" + code + ".traineddata")) {
            copy(in, temporary);
            finishModelFile(temporary, target);
            return;
        } catch (IOException bundledMissing) {
            Log.i(TAG, "OCR language is not bundled, downloading official tessdata_fast: " + code);
        }

        String remote = "https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/" +
                TESSDATA_COMMIT + "/" + code + ".traineddata";
        HttpURLConnection connection = (HttpURLConnection) new URL(remote).openConnection();
        connection.setConnectTimeout(15_000);
        connection.setReadTimeout(60_000);
        connection.setInstanceFollowRedirects(true);
        connection.setRequestProperty("User-Agent", "RTranslator-Lite11-OCR");
        try {
            int status = connection.getResponseCode();
            if (status < 200 || status >= 300) {
                throw new IOException("OCR model download failed: HTTP " + status + " for " + code);
            }
            try (InputStream in = connection.getInputStream()) {
                copy(in, temporary);
            }
            finishModelFile(temporary, target);
        } finally {
            connection.disconnect();
        }
    }

    private static void finishModelFile(File temporary, File target) throws IOException {
        if (!temporary.isFile() || temporary.length() < MIN_TRAINEDDATA_BYTES) {
            temporary.delete();
            throw new IOException("Invalid OCR traineddata: " + target.getName());
        }
        if (target.exists() && !target.delete()) {
            temporary.delete();
            throw new IOException("Cannot replace OCR traineddata: " + target.getName());
        }
        if (!temporary.renameTo(target)) {
            try (InputStream in = new java.io.FileInputStream(temporary)) {
                copy(in, target);
            }
            temporary.delete();
        }
    }

    private static void copy(InputStream in, File target) throws IOException {
        try (OutputStream out = new FileOutputStream(target)) {
            byte[] buffer = new byte[64 * 1024];
            int read;
            while ((read = in.read(buffer)) >= 0) {
                if (read > 0) out.write(buffer, 0, read);
            }
            out.flush();
        }
    }

    private static String resolveLanguageSpec(String sourceLanguageCode) {
        String normalized = normalize(sourceLanguageCode);
        Map<String, String> map = languageMap();
        String tess = map.get(normalized);
        if (tess == null) tess = "eng";
        if ("eng".equals(tess)) return "eng";
        return tess + "+eng";
    }

    private static String normalize(String code) {
        if (code == null) return "en";
        String out = code.trim().toLowerCase(Locale.ROOT);
        int dash = out.indexOf('-');
        if (dash > 0) out = out.substring(0, dash);
        int underscore = out.indexOf('_');
        if (underscore > 0) out = out.substring(0, underscore);
        return out;
    }

    private static Map<String, String> languageMap() {
        Map<String, String> m = new HashMap<>();
        m.put("ar", "ara"); m.put("bg", "bul"); m.put("ca", "cat");
        m.put("cs", "ces"); m.put("da", "dan"); m.put("de", "deu");
        m.put("el", "ell"); m.put("en", "eng"); m.put("es", "spa");
        m.put("fa", "fas"); m.put("fi", "fin"); m.put("fr", "fra");
        m.put("he", "heb"); m.put("hi", "hin"); m.put("hr", "hrv");
        m.put("hu", "hun"); m.put("id", "ind"); m.put("it", "ita");
        m.put("ja", "jpn"); m.put("kk", "kaz"); m.put("ko", "kor");
        m.put("mk", "mkd"); m.put("ms", "msa"); m.put("nl", "nld");
        m.put("no", "nor"); m.put("pl", "pol"); m.put("pt", "por");
        m.put("ro", "ron"); m.put("ru", "rus"); m.put("sk", "slk");
        m.put("sl", "slv"); m.put("sr", "srp"); m.put("sv", "swe");
        m.put("ta", "tam"); m.put("th", "tha"); m.put("tr", "tur");
        m.put("uk", "ukr"); m.put("ur", "urd"); m.put("uz", "uzb");
        m.put("vi", "vie"); m.put("zh", "chi_sim"); m.put("yue", "chi_tra");
        return m;
    }

    private static String clean(String text) {
        if (text == null) return "";
        return text.replace("\r\n", "\n")
                .replaceAll("[ \\t]+\\n", "\\n")
                .replaceAll("\\n{4,}", "\\n\\n\\n")
                .trim();
    }
}
''', encoding="utf-8")

# ---------------------------------------------------------------------------
# Dedicated Documents / OCR activity.
# ---------------------------------------------------------------------------
(java_dir / "DocumentsActivity.java").write_text(r'''package nie.translator.rtranslator.documents;

import android.content.ClipData;
import android.content.ClipboardManager;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.graphics.Bitmap;
import android.graphics.ImageDecoder;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Size;
import android.view.View;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AlertDialog;
import androidx.core.content.FileProvider;

import com.google.android.material.button.MaterialButton;

import org.xmlpull.v1.XmlPullParser;
import org.xmlpull.v1.XmlPullParserFactory;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

import nie.translator.rtranslator.GeneralActivity;
import nie.translator.rtranslator.Global;
import nie.translator.rtranslator.R;
import nie.translator.rtranslator.tools.CustomLocale;
import nie.translator.rtranslator.voice_translation.neural_networks.translation.Translator;

/** Lite11 standalone document OCR/import/translation screen. */
public class DocumentsActivity extends GeneralActivity {
    private static final int REQ_IMAGE = 4101;
    private static final int REQ_PDF = 4102;
    private static final int REQ_DOCX = 4103;
    private static final int REQ_CAMERA = 4104;
    private static final int MAX_BITMAP_DIMENSION = 2600;
    private static final int TRANSLATION_CHUNK_CHARS = 1600;

    private Global global;
    private CustomLocale sourceLanguage;
    private CustomLocale targetLanguage;
    private Uri pendingCameraUri;
    private final ExecutorService worker = Executors.newSingleThreadExecutor();

    private TextView status;
    private TextView modelText;
    private MaterialButton sourceButton;
    private MaterialButton targetButton;
    private MaterialButton translateButton;
    private ProgressBar progress;
    private EditText originalText;
    private EditText translatedText;
    private ImageView preview;

    private volatile boolean busy = false;

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_documents);
        global = (Global) getApplication();
        sourceLanguage = global.getFirstTextLanguage(true);
        targetLanguage = global.getSecondTextLanguage(true);

        status = findViewById(R.id.documents_status);
        modelText = findViewById(R.id.documents_model);
        sourceButton = findViewById(R.id.documents_source_language);
        targetButton = findViewById(R.id.documents_target_language);
        translateButton = findViewById(R.id.documents_translate);
        progress = findViewById(R.id.documents_progress);
        originalText = findViewById(R.id.documents_original_text);
        translatedText = findViewById(R.id.documents_translated_text);
        preview = findViewById(R.id.documents_preview);

        findViewById(R.id.documents_back).setOnClickListener(v -> finish());
        findViewById(R.id.documents_camera).setOnClickListener(v -> takePhoto());
        findViewById(R.id.documents_photo).setOnClickListener(v -> pickImage());
        findViewById(R.id.documents_pdf).setOnClickListener(v -> pickPdf());
        findViewById(R.id.documents_docx).setOnClickListener(v -> pickDocx());
        findViewById(R.id.documents_swap).setOnClickListener(v -> swapLanguages());
        findViewById(R.id.documents_clear).setOnClickListener(v -> clearDocument());
        findViewById(R.id.documents_copy_original).setOnClickListener(v -> copyText(originalText.getText().toString()));
        findViewById(R.id.documents_copy_translation).setOnClickListener(v -> copyText(translatedText.getText().toString()));
        findViewById(R.id.documents_share_translation).setOnClickListener(v -> shareTranslation());
        sourceButton.setOnClickListener(v -> showLanguagePicker(true));
        targetButton.setOnClickListener(v -> showLanguagePicker(false));
        translateButton.setOnClickListener(v -> translateDocument());

        updateLanguageUi();
        setStatus("Choose Camera, Photo, PDF or DOCX", false);
    }

    @Override
    protected void onDestroy() {
        worker.shutdownNow();
        super.onDestroy();
    }

    private void takePhoto() {
        if (busy) return;
        try {
            File dir = new File(getCacheDir(), "temporary_images");
            if (!dir.exists() && !dir.mkdirs()) throw new IOException("Cannot create camera cache");
            File file = new File(dir, "ocr_" + System.currentTimeMillis() + ".jpg");
            pendingCameraUri = FileProvider.getUriForFile(this,
                    "com.gallery.RTranslator.lite11.provider", file);
            Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
            intent.putExtra(MediaStore.EXTRA_OUTPUT, pendingCameraUri);
            intent.addFlags(Intent.FLAG_GRANT_WRITE_URI_PERMISSION | Intent.FLAG_GRANT_READ_URI_PERMISSION);
            startActivityForResult(intent, REQ_CAMERA);
        } catch (Exception e) {
            showError("Camera: " + safeMessage(e));
        }
    }

    private void pickImage() {
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("image/*");
        startActivityForResult(intent, REQ_IMAGE);
    }

    private void pickPdf() {
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("application/pdf");
        startActivityForResult(intent, REQ_PDF);
    }

    private void pickDocx() {
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("application/vnd.openxmlformats-officedocument.wordprocessingml.document");
        startActivityForResult(intent, REQ_DOCX);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode != RESULT_OK) return;
        if (requestCode == REQ_CAMERA && pendingCameraUri != null) {
            importImage(pendingCameraUri, "Camera photo");
            return;
        }
        if (data == null || data.getData() == null) return;
        Uri uri = data.getData();
        if (requestCode == REQ_IMAGE) importImage(uri, "Photo");
        else if (requestCode == REQ_PDF) importPdf(uri);
        else if (requestCode == REQ_DOCX) importDocx(uri);
    }

    private void importImage(Uri uri, String label) {
        startBusy("Loading image…");
        worker.execute(() -> {
            Bitmap bitmap = null;
            try {
                bitmap = decodeBitmap(uri);
                Bitmap finalBitmap = bitmap;
                runOnUiThread(() -> {
                    preview.setImageBitmap(finalBitmap);
                    preview.setVisibility(View.VISIBLE);
                });
                postStatus("Running OCR…");
                String text = OcrEngine.recognizeBitmap(this, bitmap, sourceLanguage.getCode());
                String finalText = text;
                runOnUiThread(() -> {
                    originalText.setText(finalText);
                    translatedText.setText("");
                    finishBusy(label + ": OCR complete · " + finalText.length() + " characters");
                });
            } catch (Throwable t) {
                if (bitmap != null) bitmap.recycle();
                postError("OCR failed: " + safeMessage(t));
            }
        });
    }

    private void importPdf(Uri uri) {
        startBusy("Opening PDF…");
        preview.setVisibility(View.GONE);
        worker.execute(() -> {
            try {
                String text = OcrEngine.recognizePdf(this, uri, sourceLanguage.getCode(),
                        (current, total, message) -> postStatus(message));
                runOnUiThread(() -> {
                    originalText.setText(text);
                    translatedText.setText("");
                    finishBusy("PDF OCR complete · " + text.length() + " characters");
                });
            } catch (Throwable t) {
                postError("PDF OCR failed: " + safeMessage(t));
            }
        });
    }

    private void importDocx(Uri uri) {
        startBusy("Reading DOCX…");
        preview.setVisibility(View.GONE);
        worker.execute(() -> {
            try {
                String text = extractDocxText(uri);
                runOnUiThread(() -> {
                    originalText.setText(text);
                    translatedText.setText("");
                    finishBusy("DOCX text extracted · " + text.length() + " characters");
                });
            } catch (Throwable t) {
                postError("DOCX import failed: " + safeMessage(t));
            }
        });
    }

    private Bitmap decodeBitmap(Uri uri) throws IOException {
        ImageDecoder.Source source = ImageDecoder.createSource(getContentResolver(), uri);
        Bitmap decoded = ImageDecoder.decodeBitmap(source, (decoder, info, src) -> {
            decoder.setAllocator(ImageDecoder.ALLOCATOR_SOFTWARE);
            Size size = info.getSize();
            int max = Math.max(size.getWidth(), size.getHeight());
            if (max > MAX_BITMAP_DIMENSION) {
                int sample = Math.max(1, (int) Math.ceil(max / (double) MAX_BITMAP_DIMENSION));
                decoder.setTargetSampleSize(sample);
            }
        });
        if (decoded.getConfig() == Bitmap.Config.HARDWARE) {
            Bitmap software = decoded.copy(Bitmap.Config.ARGB_8888, false);
            decoded.recycle();
            return software;
        }
        return decoded;
    }

    private String extractDocxText(Uri uri) throws Exception {
        byte[] documentXml = null;
        try (InputStream raw = getContentResolver().openInputStream(uri)) {
            if (raw == null) throw new IOException("Cannot open DOCX");
            try (ZipInputStream zip = new ZipInputStream(raw)) {
                ZipEntry entry;
                while ((entry = zip.getNextEntry()) != null) {
                    if ("word/document.xml".equals(entry.getName())) {
                        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
                        byte[] buffer = new byte[32 * 1024];
                        int read;
                        while ((read = zip.read(buffer)) >= 0) {
                            if (read > 0) bytes.write(buffer, 0, read);
                        }
                        documentXml = bytes.toByteArray();
                        break;
                    }
                }
            }
        }
        if (documentXml == null) throw new IOException("word/document.xml not found");

        XmlPullParserFactory factory = XmlPullParserFactory.newInstance();
        factory.setNamespaceAware(true);
        XmlPullParser parser = factory.newPullParser();
        parser.setInput(new ByteArrayInputStream(documentXml), StandardCharsets.UTF_8.name());
        StringBuilder out = new StringBuilder();
        int event = parser.getEventType();
        while (event != XmlPullParser.END_DOCUMENT) {
            if (event == XmlPullParser.START_TAG) {
                String name = parser.getName();
                if ("t".equals(name)) {
                    String value = parser.nextText();
                    if (value != null) out.append(value);
                } else if ("tab".equals(name)) {
                    out.append('\t');
                } else if ("br".equals(name) || "cr".equals(name)) {
                    out.append('\n');
                }
            } else if (event == XmlPullParser.END_TAG && "p".equals(parser.getName())) {
                out.append('\n');
            }
            event = parser.next();
        }
        return out.toString().replaceAll("[ \\t]+\\n", "\\n")
                .replaceAll("\\n{3,}", "\\n\\n").trim();
    }

    private void translateDocument() {
        if (busy) return;
        String text = originalText.getText().toString().trim();
        if (text.isEmpty()) {
            Toast.makeText(this, "Import or enter text first", Toast.LENGTH_SHORT).show();
            return;
        }
        final List<String> chunks = chunkText(text, TRANSLATION_CHUNK_CHARS);
        startBusy("Preparing translation…");
        translatedText.setText("");
        global.initializeTranslator(new Translator.GeneralListener() {
            @Override
            public void onSuccess() {
                translateChunk(chunks, 0, new StringBuilder());
            }

            @Override
            public void onFailure(int[] reasons, long value) {
                postError("Translation model is not ready");
            }
        });
    }

    private void translateChunk(List<String> chunks, int index, StringBuilder result) {
        if (index >= chunks.size()) {
            runOnUiThread(() -> {
                translatedText.setText(result.toString().trim());
                finishBusy("Translation complete · " + chunks.size() + " chunk(s)");
            });
            return;
        }
        postStatus("Translating " + (index + 1) + " / " + chunks.size() + "…");
        String chunk = chunks.get(index);
        global.getTranslator().translate(chunk, sourceLanguage, targetLanguage, global.getBeamSize(), false,
                new Translator.TranslateListener() {
                    private boolean finished = false;

                    @Override
                    public void onTranslatedText(String textToTranslate, String text, String[] synonyms,
                                                 long resultID, boolean isFinal, ResultType resultType,
                                                 CustomLocale languageOfText) {
                        if (!isFinal || finished) return;
                        finished = true;
                        if (result.length() > 0) result.append("\n\n");
                        result.append(text.trim());
                        translateChunk(chunks, index + 1, result);
                    }

                    @Override
                    public void onFailure(int[] reasons, long value) {
                        if (finished) return;
                        finished = true;
                        postError("Translation failed at chunk " + (index + 1));
                    }
                });
    }

    private static List<String> chunkText(String text, int limit) {
        List<String> chunks = new ArrayList<>();
        String normalized = text.replace("\r\n", "\n").trim();
        String[] paragraphs = normalized.split("\\n\\s*\\n");
        StringBuilder current = new StringBuilder();
        for (String paragraph : paragraphs) {
            String p = paragraph.trim();
            if (p.isEmpty()) continue;
            if (p.length() > limit) {
                if (current.length() > 0) {
                    chunks.add(current.toString());
                    current.setLength(0);
                }
                splitLongParagraph(p, limit, chunks);
            } else if (current.length() == 0) {
                current.append(p);
            } else if (current.length() + 2 + p.length() <= limit) {
                current.append("\n\n").append(p);
            } else {
                chunks.add(current.toString());
                current.setLength(0);
                current.append(p);
            }
        }
        if (current.length() > 0) chunks.add(current.toString());
        if (chunks.isEmpty() && !normalized.isEmpty()) chunks.add(normalized);
        return chunks;
    }

    private static void splitLongParagraph(String paragraph, int limit, List<String> out) {
        int start = 0;
        while (start < paragraph.length()) {
            int end = Math.min(paragraph.length(), start + limit);
            if (end < paragraph.length()) {
                int breakAt = paragraph.lastIndexOf(' ', end);
                if (breakAt > start + limit / 2) end = breakAt;
            }
            out.add(paragraph.substring(start, end).trim());
            start = end;
            while (start < paragraph.length() && Character.isWhitespace(paragraph.charAt(start))) start++;
        }
    }

    private void showLanguagePicker(boolean source) {
        ArrayList<CustomLocale> languages = global.getTranslatorLanguages(true);
        if (languages == null || languages.isEmpty()) return;
        String[] names = new String[languages.size()];
        for (int i = 0; i < languages.size(); i++) {
            Locale locale = languages.get(i).getLocale();
            String name = locale.getDisplayLanguage(locale);
            if (name == null || name.trim().isEmpty()) name = languages.get(i).getCode();
            names[i] = name;
        }
        new AlertDialog.Builder(this)
                .setTitle(source ? "Source language / OCR" : "Target language")
                .setItems(names, (dialog, which) -> {
                    if (source) sourceLanguage = languages.get(which);
                    else targetLanguage = languages.get(which);
                    updateLanguageUi();
                })
                .show();
    }

    private void swapLanguages() {
        CustomLocale temp = sourceLanguage;
        sourceLanguage = targetLanguage;
        targetLanguage = temp;
        updateLanguageUi();
    }

    private void updateLanguageUi() {
        sourceButton.setText(displayName(sourceLanguage));
        targetButton.setText(displayName(targetLanguage));
        int mode = selectedModelMode();
        modelText.setText("Translation model: " + modelName(mode) + " · OCR follows source language");
    }

    private int selectedModelMode() {
        if (global.getTranslator() != null) return global.getTranslator().getMode();
        SharedPreferences preferences = getSharedPreferences("default", Context.MODE_PRIVATE);
        return preferences.getInt("selectedTranslationModel", Translator.MADLAD_CACHE);
    }

    private static String modelName(int mode) {
        if (mode == Translator.MILMMT) return "MiLMMT-46-1B";
        if (mode == Translator.TINY_AYA) return "Tiny Aya Global";
        if (mode == Translator.TRANSLATE_GEMMA) return "TranslateGemma 4B";
        if (mode == Translator.MADLAD || mode == Translator.MADLAD_CACHE) return "MADLAD-400 3B";
        return "model " + mode;
    }

    private static String displayName(CustomLocale language) {
        if (language == null) return "?";
        Locale locale = language.getLocale();
        String value = locale.getDisplayLanguage(locale);
        if (value == null || value.trim().isEmpty()) value = language.getCode();
        return value;
    }

    private void copyText(String text) {
        if (text == null || text.trim().isEmpty()) return;
        ClipboardManager manager = (ClipboardManager) getSystemService(CLIPBOARD_SERVICE);
        manager.setPrimaryClip(ClipData.newPlainText("RTranslator document", text));
        Toast.makeText(this, "Copied", Toast.LENGTH_SHORT).show();
    }

    private void shareTranslation() {
        String text = translatedText.getText().toString().trim();
        if (text.isEmpty()) return;
        Intent intent = new Intent(Intent.ACTION_SEND);
        intent.setType("text/plain");
        intent.putExtra(Intent.EXTRA_TEXT, text);
        startActivity(Intent.createChooser(intent, "Share translation"));
    }

    private void clearDocument() {
        if (busy) return;
        originalText.setText("");
        translatedText.setText("");
        preview.setImageDrawable(null);
        preview.setVisibility(View.GONE);
        setStatus("Choose Camera, Photo, PDF or DOCX", false);
    }

    private void startBusy(String text) {
        busy = true;
        progress.setVisibility(View.VISIBLE);
        translateButton.setEnabled(false);
        sourceButton.setEnabled(false);
        targetButton.setEnabled(false);
        setStatus(text, true);
    }

    private void finishBusy(String text) {
        busy = false;
        progress.setVisibility(View.GONE);
        translateButton.setEnabled(true);
        sourceButton.setEnabled(true);
        targetButton.setEnabled(true);
        setStatus(text, false);
    }

    private void postStatus(String text) {
        runOnUiThread(() -> setStatus(text, true));
    }

    private void postError(String text) {
        runOnUiThread(() -> {
            busy = false;
            progress.setVisibility(View.GONE);
            translateButton.setEnabled(true);
            sourceButton.setEnabled(true);
            targetButton.setEnabled(true);
            setStatus(text, false);
            Toast.makeText(this, text, Toast.LENGTH_LONG).show();
        });
    }

    private void showError(String text) {
        setStatus(text, false);
        Toast.makeText(this, text, Toast.LENGTH_LONG).show();
    }

    private void setStatus(String text, boolean active) {
        status.setText(text);
    }

    private static String safeMessage(Throwable t) {
        String value = t.getMessage();
        return value == null || value.trim().isEmpty() ? t.getClass().getSimpleName() : value;
    }
}
''', encoding="utf-8")

# ---------------------------------------------------------------------------
# New UI: dedicated Documents & OCR menu.
# ---------------------------------------------------------------------------
(layout_dir / "activity_documents.xml").write_text(r'''<?xml version="1.0" encoding="utf-8"?>
<LinearLayout xmlns:android="http://schemas.android.com/apk/res/android"
    xmlns:app="http://schemas.android.com/apk/res-auto"
    android:layout_width="match_parent"
    android:layout_height="match_parent"
    android:orientation="vertical"
    android:background="@color/very_very_light_gray"
    android:fitsSystemWindows="true">

    <LinearLayout
        android:layout_width="match_parent"
        android:layout_height="64dp"
        android:gravity="center_vertical"
        android:orientation="horizontal"
        android:paddingStart="12dp"
        android:paddingEnd="12dp"
        android:background="@color/primary_very_lite">

        <androidx.appcompat.widget.AppCompatImageButton
            android:id="@+id/documents_back"
            android:layout_width="48dp"
            android:layout_height="48dp"
            android:backgroundTint="@android:color/transparent"
            android:tint="@color/primary_very_dark"
            app:srcCompat="@drawable/lite11_back_icon" />

        <TextView
            android:layout_width="0dp"
            android:layout_height="wrap_content"
            android:layout_weight="1"
            android:layout_marginStart="8dp"
            android:fontFamily="@font/nunito_sans_semibold"
            android:text="Documents &amp; OCR"
            android:textColor="@color/primary_very_dark"
            android:textSize="22sp"
            android:textStyle="bold" />
    </LinearLayout>

    <ScrollView
        android:layout_width="match_parent"
        android:layout_height="0dp"
        android:layout_weight="1"
        android:fillViewport="true">

        <LinearLayout
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:orientation="vertical"
            android:padding="20dp">

            <TextView
                android:id="@+id/documents_model"
                android:layout_width="match_parent"
                android:layout_height="wrap_content"
                android:textColor="@color/gray"
                android:textSize="14sp"
                android:paddingBottom="10dp" />

            <LinearLayout
                android:layout_width="match_parent"
                android:layout_height="wrap_content"
                android:orientation="horizontal"
                android:gravity="center_vertical">

                <com.google.android.material.button.MaterialButton
                    android:id="@+id/documents_source_language"
                    android:layout_width="0dp"
                    android:layout_height="wrap_content"
                    android:layout_weight="1"
                    android:textAllCaps="false" />

                <com.google.android.material.button.MaterialButton
                    android:id="@+id/documents_swap"
                    style="@style/Widget.MaterialComponents.Button.TextButton"
                    android:layout_width="56dp"
                    android:layout_height="wrap_content"
                    android:text="⇄"
                    android:textSize="22sp" />

                <com.google.android.material.button.MaterialButton
                    android:id="@+id/documents_target_language"
                    android:layout_width="0dp"
                    android:layout_height="wrap_content"
                    android:layout_weight="1"
                    android:textAllCaps="false" />
            </LinearLayout>

            <TextView
                android:layout_width="match_parent"
                android:layout_height="wrap_content"
                android:layout_marginTop="18dp"
                android:layout_marginBottom="8dp"
                android:text="Import"
                android:textColor="@color/primary_very_dark"
                android:textSize="18sp"
                android:textStyle="bold" />

            <LinearLayout
                android:layout_width="match_parent"
                android:layout_height="wrap_content"
                android:orientation="horizontal">

                <com.google.android.material.button.MaterialButton
                    android:id="@+id/documents_camera"
                    android:layout_width="0dp"
                    android:layout_height="wrap_content"
                    android:layout_weight="1"
                    android:layout_marginEnd="5dp"
                    android:text="Camera"
                    android:textAllCaps="false" />

                <com.google.android.material.button.MaterialButton
                    android:id="@+id/documents_photo"
                    android:layout_width="0dp"
                    android:layout_height="wrap_content"
                    android:layout_weight="1"
                    android:layout_marginStart="5dp"
                    android:text="Photo"
                    android:textAllCaps="false" />
            </LinearLayout>

            <LinearLayout
                android:layout_width="match_parent"
                android:layout_height="wrap_content"
                android:orientation="horizontal">

                <com.google.android.material.button.MaterialButton
                    android:id="@+id/documents_pdf"
                    android:layout_width="0dp"
                    android:layout_height="wrap_content"
                    android:layout_weight="1"
                    android:layout_marginEnd="5dp"
                    android:text="PDF"
                    android:textAllCaps="false" />

                <com.google.android.material.button.MaterialButton
                    android:id="@+id/documents_docx"
                    android:layout_width="0dp"
                    android:layout_height="wrap_content"
                    android:layout_weight="1"
                    android:layout_marginStart="5dp"
                    android:text="DOCX"
                    android:textAllCaps="false" />
            </LinearLayout>

            <LinearLayout
                android:layout_width="match_parent"
                android:layout_height="wrap_content"
                android:gravity="center_vertical"
                android:orientation="horizontal"
                android:paddingTop="8dp"
                android:paddingBottom="8dp">

                <ProgressBar
                    android:id="@+id/documents_progress"
                    android:layout_width="28dp"
                    android:layout_height="28dp"
                    android:visibility="gone" />

                <TextView
                    android:id="@+id/documents_status"
                    android:layout_width="0dp"
                    android:layout_height="wrap_content"
                    android:layout_weight="1"
                    android:layout_marginStart="10dp"
                    android:textColor="@color/gray"
                    android:textSize="14sp" />
            </LinearLayout>

            <ImageView
                android:id="@+id/documents_preview"
                android:layout_width="match_parent"
                android:layout_height="220dp"
                android:adjustViewBounds="true"
                android:scaleType="centerInside"
                android:visibility="gone"
                android:background="@color/accent_white" />

            <TextView
                android:layout_width="match_parent"
                android:layout_height="wrap_content"
                android:layout_marginTop="14dp"
                android:text="Original / OCR text"
                android:textColor="@color/primary_very_dark"
                android:textSize="17sp"
                android:textStyle="bold" />

            <EditText
                android:id="@+id/documents_original_text"
                android:layout_width="match_parent"
                android:layout_height="180dp"
                android:layout_marginTop="6dp"
                android:background="@color/accent_white"
                android:gravity="top|start"
                android:inputType="textMultiLine|textCapSentences"
                android:padding="12dp"
                android:scrollbars="vertical"
                android:textSize="16sp" />

            <LinearLayout
                android:layout_width="match_parent"
                android:layout_height="wrap_content"
                android:orientation="horizontal"
                android:gravity="end">

                <com.google.android.material.button.MaterialButton
                    android:id="@+id/documents_copy_original"
                    style="@style/Widget.MaterialComponents.Button.TextButton"
                    android:layout_width="wrap_content"
                    android:layout_height="wrap_content"
                    android:text="Copy"
                    android:textAllCaps="false" />

                <com.google.android.material.button.MaterialButton
                    android:id="@+id/documents_clear"
                    style="@style/Widget.MaterialComponents.Button.TextButton"
                    android:layout_width="wrap_content"
                    android:layout_height="wrap_content"
                    android:text="Clear"
                    android:textAllCaps="false" />

                <com.google.android.material.button.MaterialButton
                    android:id="@+id/documents_translate"
                    android:layout_width="wrap_content"
                    android:layout_height="wrap_content"
                    android:text="Translate"
                    android:textAllCaps="false" />
            </LinearLayout>

            <TextView
                android:layout_width="match_parent"
                android:layout_height="wrap_content"
                android:layout_marginTop="10dp"
                android:text="Translation"
                android:textColor="@color/primary_very_dark"
                android:textSize="17sp"
                android:textStyle="bold" />

            <EditText
                android:id="@+id/documents_translated_text"
                android:layout_width="match_parent"
                android:layout_height="220dp"
                android:layout_marginTop="6dp"
                android:background="@color/accent_white"
                android:gravity="top|start"
                android:inputType="textMultiLine|textCapSentences"
                android:padding="12dp"
                android:scrollbars="vertical"
                android:textSize="16sp" />

            <LinearLayout
                android:layout_width="match_parent"
                android:layout_height="wrap_content"
                android:orientation="horizontal"
                android:gravity="end"
                android:paddingBottom="16dp">

                <com.google.android.material.button.MaterialButton
                    android:id="@+id/documents_copy_translation"
                    style="@style/Widget.MaterialComponents.Button.TextButton"
                    android:layout_width="wrap_content"
                    android:layout_height="wrap_content"
                    android:text="Copy translation"
                    android:textAllCaps="false" />

                <com.google.android.material.button.MaterialButton
                    android:id="@+id/documents_share_translation"
                    style="@style/Widget.MaterialComponents.Button.TextButton"
                    android:layout_width="wrap_content"
                    android:layout_height="wrap_content"
                    android:text="Share"
                    android:textAllCaps="false" />
            </LinearLayout>
        </LinearLayout>
    </ScrollView>
</LinearLayout>
''', encoding="utf-8")

(drawable_dir / "lite11_documents_icon.xml").write_text(r'''<vector xmlns:android="http://schemas.android.com/apk/res/android"
    android:width="24dp" android:height="24dp" android:viewportWidth="24" android:viewportHeight="24">
    <path android:fillColor="#FF000000" android:pathData="M6,2h8l4,4v5h-2V7h-3V4H6v16h6v2H4V2h2zM14,13h8v2h-8zM14,17h8v2h-8zM14,21h8v2h-8z" />
</vector>
''', encoding="utf-8")

(drawable_dir / "lite11_back_icon.xml").write_text(r'''<vector xmlns:android="http://schemas.android.com/apk/res/android"
    android:width="24dp" android:height="24dp" android:viewportWidth="24" android:viewportHeight="24">
    <path android:fillColor="#FF000000" android:pathData="M20,11H7.83l5.59,-5.59L12,4l-8,8 8,8 1.42,-1.41L7.83,13H20v-2z" />
</vector>
''', encoding="utf-8")

# ---------------------------------------------------------------------------
# Add a document icon beside Settings on the main Translation toolbar.
# ---------------------------------------------------------------------------
tj = translation_fragment.read_text(encoding="utf-8")
if 'import nie.translator.rtranslator.documents.DocumentsActivity;' not in tj:
    import_needle = 'import nie.translator.rtranslator.settings.SettingsActivity;\n'
    if import_needle not in tj:
        raise SystemExit("SettingsActivity import missing")
    tj = tj.replace(import_needle, import_needle + 'import nie.translator.rtranslator.documents.DocumentsActivity;\n', 1)

field_needle = '    private AppCompatImageButton settingsButton;\n    private AppCompatImageButton settingsButtonReduced;\n'
field_replacement = ('    private AppCompatImageButton settingsButton;\n'
                     '    private AppCompatImageButton settingsButtonReduced;\n'
                     '    private AppCompatImageButton documentsButton;\n'
                     '    private AppCompatImageButton documentsButtonReduced;\n')
if field_needle not in tj:
    raise SystemExit("Translation toolbar fields missing")
tj = tj.replace(field_needle, field_replacement, 1)

bind_needle = ('        settingsButton = view.findViewById(R.id.settingsButton);\n'
               '        settingsButtonReduced = view.findViewById(R.id.settingsButton2);\n')
bind_replacement = bind_needle + ('        documentsButton = view.findViewById(R.id.documentsButton);\n'
                                  '        documentsButtonReduced = view.findViewById(R.id.documentsButton2);\n')
if bind_needle not in tj:
    raise SystemExit("Translation toolbar binding missing")
tj = tj.replace(bind_needle, bind_replacement, 1)

listener_needle = '''        settingsButton.setOnClickListener(new View.OnClickListener() {\n            @Override\n            public void onClick(View v) {\n                Intent intent = new Intent(activity, SettingsActivity.class);\n                intent.setFlags(Intent.FLAG_ACTIVITY_NEW_TASK);\n                startActivity(intent);\n            }\n        });\n'''
if listener_needle not in tj:
    raise SystemExit("Settings button listener missing")
doc_listener = '''        documentsButton.setOnClickListener(v -> startActivity(new Intent(activity, DocumentsActivity.class)));\n        documentsButtonReduced.setOnClickListener(v -> startActivity(new Intent(activity, DocumentsActivity.class)));\n'''
tj = tj.replace(listener_needle, listener_needle + doc_listener, 1)
translation_fragment.write_text(tj, encoding="utf-8")

# Main full toolbar: insert Documents button immediately before Settings.
tl = translation_layout.read_text(encoding="utf-8")
settings_xml = '''        <androidx.appcompat.widget.AppCompatImageButton\n            android:id="@+id/settingsButton"'''
if settings_xml not in tl:
    raise SystemExit("settingsButton XML missing")
doc_xml = '''        <androidx.appcompat.widget.AppCompatImageButton\n            android:id="@+id/documentsButton"\n            android:layout_width="wrap_content"\n            android:layout_height="wrap_content"\n            android:layout_marginEnd="52dp"\n            android:backgroundTint="@android:color/transparent"\n            android:tint="@color/primary_very_dark"\n            android:contentDescription="Documents and OCR"\n            app:layout_constraintBottom_toBottomOf="parent"\n            app:layout_constraintEnd_toEndOf="@+id/toolbarTranslator"\n            app:layout_constraintTop_toTopOf="parent"\n            app:srcCompat="@drawable/lite11_documents_icon" />\n\n'''
tl = tl.replace(settings_xml, doc_xml + settings_xml, 1)

# Reduced toolbar button beside settingsButton2.
settings2_xml = '''        <androidx.appcompat.widget.AppCompatImageButton\n            android:id="@+id/settingsButton2"'''
if settings2_xml not in tl:
    raise SystemExit("settingsButton2 XML missing")
doc2_xml = '''        <androidx.appcompat.widget.AppCompatImageButton\n            android:id="@+id/documentsButton2"\n            android:layout_width="wrap_content"\n            android:layout_height="wrap_content"\n            android:alpha="0"\n            android:backgroundTint="@android:color/transparent"\n            android:clickable="false"\n            android:tint="@color/primary_very_dark"\n            android:contentDescription="Documents and OCR"\n            app:layout_constraintBottom_toBottomOf="parent"\n            app:layout_constraintEnd_toStartOf="@+id/settingsButton2"\n            app:layout_constraintTop_toTopOf="parent"\n            app:srcCompat="@drawable/lite11_documents_icon" />\n\n'''
tl = tl.replace(settings2_xml, doc2_xml + settings2_xml, 1)
translation_layout.write_text(tl, encoding="utf-8")

# ---------------------------------------------------------------------------
# Manifest, dependency and separate Lite11 install identity.
# ---------------------------------------------------------------------------
m = manifest.read_text(encoding="utf-8")
activity_anchor = '''        <activity\n            android:name="nie.translator.rtranslator.settings.SettingsActivity"\n            android:exported="true"\n            android:launchMode="singleTask"\n            android:theme="@style/Theme.Settings" />\n'''
if activity_anchor not in m:
    raise SystemExit("Settings activity manifest anchor missing")
doc_activity = '''        <activity\n            android:name="nie.translator.rtranslator.documents.DocumentsActivity"\n            android:exported="false"\n            android:screenOrientation="unspecified"\n            android:theme="@style/Theme.Speech" />\n'''
m = m.replace(activity_anchor, activity_anchor + doc_activity, 1)
manifest.write_text(m, encoding="utf-8")

b = build.read_text(encoding="utf-8")
if 'applicationId "nie.translator.rtranslator.lite10"' not in b:
    raise SystemExit("Lite10 applicationId missing")
b = b.replace('applicationId "nie.translator.rtranslator.lite10"',
              'applicationId "nie.translator.rtranslator.lite11"', 1)
b = b.replace('versionCode 30012', 'versionCode 30013', 1)
b = b.replace("versionName '3.0.0-alpha3-lite10'", "versionName '3.0.0-alpha3-lite11'", 1)

dep_anchor = "implementation files('libs/llama-android-lite10.aar')"
if dep_anchor not in b:
    raise SystemExit("Lite10 llama AAR dependency missing")
b = b.replace(dep_anchor, dep_anchor + "\n    implementation 'cz.adaptech.tesseract4android:tesseract4android:4.9.0'", 1)

# Ensure dependencies that bundle native code only package the phone ABI.
if "ndk {\n            abiFilters 'arm64-v8a'\n        }" not in b:
    ndk_anchor = '        minSdkVersion 28\n'
    if ndk_anchor not in b:
        raise SystemExit("minSdk anchor missing")
    b = b.replace(ndk_anchor, ndk_anchor + "        ndk {\n            abiFilters 'arm64-v8a'\n        }\n", 1)
build.write_text(b, encoding="utf-8")

changed_labels = 0
for strings_xml in (ROOT / "app/src/main/res").glob("values*/strings.xml"):
    s = strings_xml.read_text(encoding="utf-8")
    if "RTranslator Lite10" in s:
        strings_xml.write_text(s.replace("RTranslator Lite10", "RTranslator Lite11"), encoding="utf-8")
        changed_labels += 1
if changed_labels == 0:
    raise SystemExit("Lite10 app label missing")

for p in provider_files:
    s = p.read_text(encoding="utf-8")
    if "com.gallery.RTranslator.lite10.provider" not in s:
        raise SystemExit(f"Lite10 provider missing in {p}")
    p.write_text(s.replace("com.gallery.RTranslator.lite10.provider",
                           "com.gallery.RTranslator.lite11.provider"), encoding="utf-8")

for p in (ROOT / "app/src/main/java").rglob("*.java"):
    s = p.read_text(encoding="utf-8")
    if "RTranslatorLite10" in s:
        p.write_text(s.replace("RTranslatorLite10", "RTranslatorLite11"), encoding="utf-8")

print("Lite11 Documents/OCR patch complete:")
print(" - separate Documents & OCR activity and main toolbar entry")
print(" - Camera / Photo / PDF OCR via Tesseract 5")
print(" - DOCX direct XML text extraction")
print(" - editable OCR text and translated result")
print(" - source/target language pickers, swap, copy/share/clear")
print(" - long document translation chunking through currently selected MT model")
print(" - applicationId nie.translator.rtranslator.lite11")
