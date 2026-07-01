"""
Custom Faster-Whisper Server / 自定义 Faster-Whisper 服务
- GPU/CPU dynamic switching / 支持 GPU/CPU 动态切换
- Auto-unload models with configurable idle timeout / 支持按需自动卸载模型（可配置空闲超时）
- OpenAI /v1/audio/transcriptions compatible / API 兼容 OpenAI /v1/audio/transcriptions 规范
- Load models from local whisper-models directory / 支持从本地 whisper-models 目录加载模型
- stable-ts support for advanced alignment / 支持 stable-ts 高级对齐功能
"""

import asyncio
import time
import gc
import os
import re
import unicodedata
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.responses import JSONResponse
from faster_whisper import WhisperModel
import stable_whisper
import filetype

# ========== Audio Format Validation / 音频格式验证 ==========
# Supported audio formats (OpenAI Whisper compatible + AAC)
# 支持的音频格式（兼容 OpenAI Whisper + AAC）
SUPPORTED_FORMATS = {
    'aac', 'flac', 'm4a', 'mp3', 'mp4', 'mpeg', 'mpga', 'oga', 'ogg', 'wav', 'webm'
}

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# 禁用 Uvicorn 的默认访问日志
logging.getLogger("uvicorn.access").handlers = []
logging.getLogger("uvicorn.access").propagate = False

# ========== Environment Variables / 环境变量配置 ==========
# GPU/CPU device selection (default: cuda, optional: cpu) / GPU/CPU 设备选择 (默认: cuda, 可选: cpu)
DEVICE = os.getenv("WHISPER_DEVICE", "cuda")
# Compute type (GPU default: float16, CPU default: int8) / 计算类型 (GPU 默认: float16, CPU 默认: int8)
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "float16" if DEVICE == "cuda" else "int8")
# CPU threads (CPU mode only) / CPU 线程数 (仅 CPU 模式有效)
THREADS = int(os.getenv("WHISPER_THREADS", "8"))
# Model cache directory / 模型缓存目录
CACHE_DIR = os.getenv("WHISPER_CACHE_DIR", "/root/.cache/huggingface/hub")
# Idle timeout in seconds, 0 means never unload (default: 600s = 10min) / 空闲超时时间（秒），0 表示永不卸载 (默认: 600秒 = 10分钟)
IDLE_TIMEOUT = int(os.getenv("WHISPER_IDLE_TIMEOUT", "600"))
# Default model / 默认模型
DEFAULT_MODEL = os.getenv("WHISPER_MODEL", "small")
# HuggingFace offline mode / HuggingFace 离线模式:
# - "0" = Fully online (check updates every time, not recommended) / 完全在线（每次检查更新，不推荐）
# - "1" = Fully offline (error if not found locally) / 完全离线（本地没有就报错）
# - "auto" = Smart mode (use local if exists, download if not, no update checks) / 智能模式（本地有就用本地，没有就下载，不检查更新）
HF_OFFLINE = os.getenv("HF_HUB_OFFLINE", "auto")

# Set environment variable (fully offline mode only) / 设置环境变量（仅在完全离线模式）
if HF_OFFLINE == "1":
    os.environ["HF_HUB_OFFLINE"] = "1"

# ========== API Key Authentication / API Key 认证 ==========
# Enable API key authentication (default: false) / 启用 API Key 认证（默认：关闭）
API_KEY_REQUIRED = os.getenv("API_KEY_REQUIRED", "false").lower() in ("true", "1", "yes", "on")
# Valid API keys (separated by semicolon) / 有效的 API Key（分号分隔）
API_KEYS = [k.strip() for k in os.getenv("API_KEYS", "").split(";") if k.strip()] if API_KEY_REQUIRED else []

# ========== Forced Alignment Language Detection / 强制对齐语言检测 ==========
# Auto-detect language for forced alignment when not specified (default: false)
# 强制对齐时自动检测语言（默认：关闭，要求用户必须指定）
ALIGNMENT_AUTO_DETECT_LANGUAGE = os.getenv("ALIGNMENT_AUTO_DETECT_LANGUAGE", "false").lower() in ("true", "1", "yes", "on")

logger.info(f"Config / 配置: DEVICE={DEVICE}, COMPUTE_TYPE={COMPUTE_TYPE}, THREADS={THREADS}")
logger.info(f"Cache dir / 缓存目录: {CACHE_DIR}")
logger.info(f"Idle timeout / 空闲超时: {IDLE_TIMEOUT}s ({'disabled / 禁用自动卸载' if IDLE_TIMEOUT == 0 else 'enabled / 启用自动卸载'})")
offline_mode = "fully offline / 完全离线" if HF_OFFLINE == "1" else ("fully online / 完全在线" if HF_OFFLINE == "0" else "smart mode (local first) / 智能模式(本地优先)")
logger.info(f"HuggingFace mode / HuggingFace 模式: {offline_mode}")
align_lang_mode = "auto-detect / 自动检测" if ALIGNMENT_AUTO_DETECT_LANGUAGE else "require user input / 要求用户指定"
logger.info(f"Alignment language detection / 强制对齐语言检测: {align_lang_mode}")

# ========== Global State Management / 全局状态管理 ==========
class ModelState:
    def __init__(self):
        self.model = None
        self.model_name = None
        self.engine_type = None  # "faster-whisper" or "stable-ts"
        self.last_used_time = 0
        self._lock = asyncio.Lock()

    def is_loaded(self):
        return self.model is not None

    def update_last_used(self):
        self.last_used_time = time.time()

    async def load(self, model_request_name: str):
        """Load specified model with engine detection / 加载指定模型（自动检测引擎类型）"""
        async with self._lock:
            # Detect engine type and base model name / 检测引擎类型和基础模型名称
            is_stable_ts = model_request_name.startswith("stable-ts-")
            base_model_name = model_request_name.replace("stable-ts-", "")
            engine_type = "stable-ts" if is_stable_ts else "faster-whisper"

            # If model exists and is different, unload first / 如果已有模型且不是请求的模型，先卸载
            if self.model is not None and (self.model_name != base_model_name or self.engine_type != engine_type):
                logger.info(f"Switching model / 切换模型: unload {self.engine_type}-{self.model_name} -> load {engine_type}-{base_model_name}")
                self._unload()

            # If not loaded, load the requested model / 如果未加载，则加载请求的模型
            if self.model is None:
                logger.info(f"Loading model / 正在加载模型: {base_model_name} ({engine_type}) to / 到 {DEVICE}...")
                try:
                    # Smart mode: check if model exists locally / 智能模式：检查本地是否已有模型
                    local_files_only = False
                    if HF_OFFLINE == "auto":
                        # Check local cache / 检查本地缓存是否存在
                        if os.path.isdir(CACHE_DIR):
                            from huggingface_hub import scan_cache_dir
                            cache = scan_cache_dir(CACHE_DIR)
                            # Check if matching model cache exists (match by suffix so non-Systran
                            # repos also work, e.g. mobiuslabsgmbh/faster-whisper-large-v3-turbo)
                            # 检查是否有匹配的模型缓存（按后缀匹配，兼容 Systran / mobiuslabsgmbh 等不同组织仓库）
                            model_suffix = f"faster-whisper-{base_model_name}"
                            cached_repos = [repo.repo_id for repo in cache.repos]
                            if any(r.endswith(model_suffix) for r in cached_repos):
                                local_files_only = True
                                logger.info(f"Model {base_model_name} exists locally, using offline mode / 本地已存在模型 {base_model_name}，使用离线模式")
                            else:
                                logger.info(f"Model {base_model_name} not found locally, will download / 本地无模型 {base_model_name}，将下载")
                        else:
                            logger.info(f"Cache dir not found, will download / 缓存目录不存在，将下载模型")
                    elif HF_OFFLINE == "1":
                        local_files_only = True

                    # Load based on engine type / 根据引擎类型加载
                    if is_stable_ts:
                        self.model = stable_whisper.load_faster_whisper(
                            base_model_name,
                            device=DEVICE,
                            compute_type=COMPUTE_TYPE if DEVICE == "cuda" else "int8",
                            cpu_threads=THREADS if DEVICE == "cpu" else 0,
                            num_workers=1,
                            download_root=CACHE_DIR,
                            local_files_only=local_files_only,
                        )
                    else:
                        self.model = WhisperModel(
                            model_size_or_path=base_model_name,
                            device=DEVICE,
                            device_index=0,
                            compute_type=COMPUTE_TYPE,
                            cpu_threads=THREADS if DEVICE == "cpu" else 0,
                            num_workers=1,
                            download_root=CACHE_DIR,
                            local_files_only=local_files_only,
                        )

                    self.model_name = base_model_name
                    self.engine_type = engine_type
                    self.last_used_time = time.time()
                    logger.info(f"Model {base_model_name} ({engine_type}) loaded successfully / 模型 {base_model_name} ({engine_type}) 加载完成")
                except Exception as e:
                    logger.error(f"Model load failed / 加载模型失败: {e}")
                    raise HTTPException(status_code=500, detail=f"Model load failed / 模型加载失败: {str(e)}")

        return self.model, self.engine_type

    def _unload(self):
        """Internal unload method (no lock) / 内部卸载方法（不加锁）"""
        if self.model is not None:
            logger.info(f"Releasing resources for model {self.model_name} ({self.engine_type})... / 释放模型 {self.model_name} ({self.engine_type}) 的系统资源...")
            del self.model
            self.model = None
            self.model_name = None
            self.engine_type = None
            gc.collect()

            # GPU memory cleanup / GPU 内存清理
            if DEVICE == "cuda":
                try:
                    import torch
                    torch.cuda.empty_cache()
                    logger.info("GPU cache cleared / GPU 缓存已清理")
                except ImportError:
                    pass

    async def unload(self):
        """Public unload method (with lock) / 公开卸载方法（加锁）"""
        async with self._lock:
            self._unload()

    def should_unload(self):
        """Check if model should be unloaded / 检查是否应该卸载"""
        if IDLE_TIMEOUT == 0:
            return False
        if self.model is None:
            return False
        idle_time = time.time() - self.last_used_time
        return idle_time > IDLE_TIMEOUT


model_state = ModelState()


# ========== Watchdog Task / 看门狗任务 ==========
async def watchdog_task():
    """Periodically check and unload idle models / 定期检查并卸载空闲模型"""
    while True:
        await asyncio.sleep(60)  # Check every minute / 每分钟检查一次
        if model_state.should_unload():
            idle_time = int(time.time() - model_state.last_used_time)
            logger.info(f"Model idle for {idle_time}s, exceeding threshold {IDLE_TIMEOUT}s, unloading / 模型空闲 {idle_time} 秒，超过阈值 {IDLE_TIMEOUT} 秒，执行卸载")
            await model_state.unload()


# ========== FastAPI Application / FastAPI 应用 ==========
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management / 应用生命周期管理"""
    # Start watchdog task / 启动看门狗任务
    watchdog = asyncio.create_task(watchdog_task())
    logger.info("Custom Faster-Whisper server started / 自定义 Faster-Whisper 服务已启动")
    yield
    # Cleanup on shutdown / 关闭时清理
    watchdog.cancel()
    await model_state.unload()
    logger.info("Server shutdown / 服务已关闭")


def _validate_audio_file(filename: str, file_content: bytes) -> tuple[bool, str | None]:
    """
    Validate audio file format using filetype library
    使用 filetype 库验证音频文件格式

    Returns: (is_valid, error_message)
    """
    # 1. Check file extension / 检查文件扩展名
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    if ext not in SUPPORTED_FORMATS:
        return False, f"Invalid file format. Supported formats: {sorted(SUPPORTED_FORMATS)}"

    # 2. Use filetype to detect real format / 使用 filetype 检测真实格式
    kind = filetype.guess(file_content)

    # No type detected / 未检测到类型
    if kind is None:
        return False, f"Unable to determine file type. The file may be empty or corrupted."

    # 3. Check if detected type is supported / 检查检测到的类型是否支持
    detected_ext = kind.extension

    # Allow format aliases / 允许格式别名
    aliases = {
        'm4a': {'aac', 'mp4'},
        'mp4': {'aac', 'm4a'},
        'oga': {'ogg'},
        'mpga': {'mp3'},
        'mpeg': {'mp3'},
    }

    # Check if detected extension matches or is an alias / 检查检测到的扩展名是否匹配或是别名
    valid_matches = {detected_ext} | aliases.get(detected_ext, set())
    if ext not in valid_matches:
        return False, f"File content is {detected_ext}, but extension is .{ext}"

    # 4. Verify it's audio or video (mp4/webm are video containers with audio) / 验证是音频或视频
    if not (kind.mime.startswith('audio/') or kind.mime.startswith('video/')):
        return False, f"Detected file type {kind.mime} is not an audio format."

    return True, None


def _sanitize_text_for_alignment(text: str, language: str = None) -> str:
    """
    Deep clean exact_text to prevent tokenization anomalies and alignment timestamp freezing.
    深度清洗 exact_text，防止 Tokenizer 解析异常和对齐时间戳卡死。

    Args:
        text: Input text to sanitize / 待清洗的输入文本
        language: Optional language code for language-specific cleaning (e.g., 'zh', 'en')
                 可选的语言代码，用于特定语言的清洗

    Returns:
        Sanitized text / 清洗后的文本
    """
    if not text:
        return text

    original_text = text

    # 1. Unicode Normalization (NFKC)
    # Convert full-width characters to half-width (１２３ -> 123)
    # Normalize composed characters (é -> é)
    # 将全角字符转为半角（如 １２３ -> 123），组合字符转为单一字符
    text = unicodedata.normalize('NFKC', text)

    # 2. Remove zero-width characters and format control characters
    # 移除常见的零宽字符和格式控制字符
    zero_width_chars = ['\u200b', '\u200c', '\u200d', '\ufeff', '\xa0']
    for char in zero_width_chars:
        text = text.replace(char, ' ')

    # 3. Remove ASCII control characters (except whitespace that will be handled next)
    # Matches: \x00-\x08, \x0B, \x0C, \x0E-\x1F, \x7F
    # 移除 ASCII 控制字符 (保留基础空白，因为下一步会处理)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    # 4. Normalize whitespace but preserve line breaks
    # Replace tabs and multiple spaces with single space, keep newlines for subtitle segmentation
    # 归一化空白符但保留换行（用于字幕分段）
    # [^\S\n] matches any whitespace except newline
    text = re.sub(r'[^\S\n]+', ' ', text)

    # 5. [Optional] Language-specific: Chinese text cleaning
    # Remove spaces between Chinese characters (helps with Whisper's Chinese tokenization)
    # 仅中文：移除中文字符之间的空格
    if language in ['zh', 'zh-CN', 'zh-TW', 'zh-HK']:
        # Match: Chinese character + space + Chinese character, remove the space
        # 匹配：中文字符 + 空格 + 中文字符，并移除空格
        text = re.sub(r'(?<=[\u4e00-\u9fa5])\s+(?=[\u4e00-\u9fa5])', '', text)

    text = text.strip()

    # Log warning if sanitization completely cleared the text
    # 如果清洗后文本变化极大（比如被清空了），记录一条 warning
    if not text and original_text:
        logger.warning("Sanitization completely cleared the text. Original text might contain only invalid characters.")

    return text


def _detect_language(model, audio_file: str) -> str:
    """
    Detect language from audio file using the loaded model
    使用加载的模型从音频文件中检测语言

    Returns: detected language code (e.g., 'en', 'zh', 'fr')
    """
    # Run a quick transcription with language detection only
    # 运行快速转录以仅检测语言
    segments, info = model.transcribe(
        audio_file,
        language=None,  # Force auto-detection / 强制自动检测
        beam_size=1,
        word_timestamps=False,
    )
    # Consume the generator to trigger detection / 消费生成器以触发检测
    _ = list(segments)
    detected = info.language if hasattr(info, 'language') else 'en'
    logger.info(f"Auto-detected language for alignment: {detected}")
    return detected


app = FastAPI(
    title="Custom Faster-Whisper Server",
    description="OpenAI API compatible Whisper speech recognition service with GPU acceleration and auto-unload / 兼容 OpenAI API 的 Whisper 语音识别服务，支持 GPU 加速和自动卸载",
    version="1.0.0",
    lifespan=lifespan
)


# ========== Custom Access Log Middleware / 自定义访问日志中间件 ==========
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log HTTP requests with custom filtering and format / 自定义 HTTP 请求日志"""
    import datetime
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time

    # Format: [YYYY-MM-DD HH:MM:SS] METHOD PATH STATUS
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Skip logging successful health checks / 跳过成功的健康检查日志
    if request.url.path == "/health" and response.status_code == 200:
        return response

    # Log the request / 记录请求
    logger.info(f'{request.client.host}:{request.client.port} - "{request.method} {request.url.path} HTTP/{request.scope["http_version"]}" {response.status_code}')

    return response


# ========== API Key Authentication Middleware / API Key 认证中间件 ==========
@app.middleware("http")
async def api_key_auth(request: Request, call_next):
    """API key authentication middleware / API Key 认证中间件

    Supports two authentication methods / 支持两种认证方式:
    1. Authorization header: Authorization: Bearer sk-xxx
    2. Query parameter: ?api_key=sk-xxx
    """
    # Public endpoints that don't require authentication / 不需要认证的公开端点
    public_paths = ["/health", "/v1/models", "/docs", "/redoc", "/openapi.json"]
    if request.url.path in public_paths:
        return await call_next(request)

    # If authentication is disabled, allow all requests / 如果关闭了认证，允许所有请求
    if not API_KEY_REQUIRED:
        return await call_next(request)

    # Extract API key from request / 从请求中提取 API Key
    api_key = None

    # Method 1: Authorization header / 方式 1：Authorization 头
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        api_key = auth_header.replace("Bearer ", "")

    # Method 2: Query parameter / 方式 2：查询参数
    if not api_key:
        api_key = request.query_params.get("api_key")

    # Validate API key / 验证 API Key
    if not api_key or api_key not in API_KEYS:
        logger.warning(f"Failed authentication attempt / 认证失败: {request.client.host}:{request.client.port} - {request.url.path}")
        return JSONResponse(
            status_code=401,
            content={"error": {"message": "Invalid API key", "type": "invalid_request_error"}}
        )

    # Authentication successful, proceed with request / 认证成功，继续处理请求
    return await call_next(request)


@app.get("/health")
async def health_check():
    """Health check endpoint / 健康检查接口"""
    return {
        "status": "healthy",
        "device": DEVICE,
        "compute_type": COMPUTE_TYPE,
        "threads": THREADS,
        "model_loaded": model_state.is_loaded(),
        "model_name": model_state.model_name,
        "engine_type": model_state.engine_type,
        "idle_timeout": IDLE_TIMEOUT,
        "hf_offline": HF_OFFLINE,
        "cache_dir": CACHE_DIR,
        "api_key_required": API_KEY_REQUIRED,
        "api_keys_configured": len(API_KEYS) if API_KEY_REQUIRED else 0
    }


@app.get("/v1/models")
async def list_models():
    """List available models (simulates OpenAI API) / 列出可用模型（模拟 OpenAI API）"""
    # Can scan local directory for actual available models / 这里可以扫描本地目录获取实际可用模型
    # Currently returns common model list / 目前返回常见模型列表
    base_models = [
        "tiny", "tiny.en", "base", "base.en", "small", "small.en",
        "medium", "medium.en", "large-v1", "large-v2", "large-v3"
    ]
    data = []
    for m in base_models:
        data.append({"id": m, "object": "model", "owned_by": "Systran"})
        data.append({"id": f"stable-ts-{m}", "object": "model", "owned_by": "Systran"})
    return {"object": "list", "data": data}


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form(DEFAULT_MODEL),
    prompt: str = Form(None),
    response_format: str = Form("json"),
    language: str = Form(None),
    temperature: float = Form(0.0),
    timestamp_granularities: list[str] = Form(None),
    exact_text: str = Form(None),
    sanitize_text: str = Form("true"),
    # VAD parameters / VAD 语音活动检测参数
    vad_filter: str = Form("false"),
    vad_min_silence_duration_ms: int = Form(500),
    vad_speech_pad_ms: int = Form(400),
    vad_threshold: float = Form(0.5),
):
    """
    OpenAI API compatible speech recognition endpoint / 兼容 OpenAI API 的语音识别接口

    Args:
        file: Audio file / 音频文件
        model: Model name (tiny, base, small, medium, large-v2, large-v3, etc.)
               Use stable-ts-* prefix for stable-ts engine (e.g., stable-ts-small)
        prompt: Initial prompt / 初始提示词
        response_format: Response format (json, verbose_json, text, srt, vtt) / 返回格式
        language: Language code (e.g., 'zh', 'en', 'ja'), None for auto-detect / 语言代码，None 表示自动检测
        temperature: Sampling temperature / 采样温度
        timestamp_granularities: Timestamp granularity (word) / 时间戳粒度
        exact_text: [Custom] Full transcript for forced alignment (requires stable-ts-* model)
                   [自定义] 完整逐字稿用于强制对齐（需要 stable-ts-* 模型）
        sanitize_text: [Custom] Enable text sanitization for forced alignment (default: true)
                      [自定义] 启用强制对齐文本清洗（默认：true）
        vad_filter: [Custom] Enable VAD voice activity detection filter (default: false)
                   [自定义] 启用 VAD 语音活动检测过滤（默认：false）
        vad_min_silence_duration_ms: [Custom] Minimum silence duration in ms to split (default: 500)
                                     [自定义] 静音最短持续时间（毫秒），超过才切割（默认：500）
        vad_speech_pad_ms: [Custom] Padding in ms around speech segments (default: 400)
                           [自定义] 语音段前后填充时间（毫秒）（默认：400）
        vad_threshold: [Custom] VAD confidence threshold (default: 0.5)
                      [自定义] VAD 置信度阈值（默认：0.5）
    """
    # Update last used time / 更新最后使用时间
    model_state.update_last_used()

    # Read and validate file / 读取并验证文件
    file_content = await file.read()
    is_valid, error_msg = _validate_audio_file(file.filename, file_content)
    if not is_valid:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": error_msg,
                    "type": "invalid_request_error"
                }
            }
        )

    # Load requested model / 加载请求的模型
    whisper_model, engine_type = await model_state.load(model)

    # Save temp file / 保存临时文件
    temp_file = f"/tmp/temp_whisper_{time.time()}_{file.filename}"
    try:
        with open(temp_file, "wb") as f:
            f.write(file_content)

        # Check for forced alignment request / 检查强制对齐请求
        if exact_text:
            if engine_type != "stable-ts":
                raise HTTPException(
                    status_code=400,
                    detail="Forced alignment (exact_text) requires a 'stable-ts-' prefixed model. "
                           f"Current engine: {engine_type}. "
                           "强制对齐 (exact_text) 需要使用 'stable-ts-' 开头的模型"
                )

            # Handle language detection for forced alignment / 处理强制对齐的语言检测
            alignment_language = language
            if not alignment_language:
                if ALIGNMENT_AUTO_DETECT_LANGUAGE:
                    # Auto-detect language from audio / 从音频自动检测语言
                    logger.info("No language specified for forced alignment, auto-detecting... / 未指定语言，自动检测中...")
                    alignment_language = _detect_language(whisper_model, temp_file)
                else:
                    # Require user to specify language / 要求用户指定语言
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "error": {
                                "message": "The 'language' parameter is required for forced alignment (exact_text). "
                                          "Please specify the language (e.g., language=en, language=zh, language=fr). "
                                          "Or set ALIGNMENT_AUTO_DETECT_LANGUAGE=true to enable auto-detection. "
                                          "强制对齐 (exact_text) 需要指定 'language' 参数。请指定语言（如：language=en、language=zh、language=fr）。"
                                          "或设置 ALIGNMENT_AUTO_DETECT_LANGUAGE=true 启用自动检测。",
                                "type": "invalid_request_error"
                            }
                        }
                    )

            # Sanitize text before alignment to prevent tokenizer crashes
            # 在对齐前清洗文本，防止 Tokenizer 崩溃
            should_sanitize = sanitize_text.lower() in ("true", "1", "yes", "on")
            if should_sanitize:
                original_len = len(exact_text)
                exact_text = _sanitize_text_for_alignment(exact_text, language=alignment_language)
                logger.info(f"Sanitized exact_text for alignment. Length: {original_len} -> {len(exact_text)} / 已清洗对齐文本。")

            # Execute forced alignment / 执行强制对齐
            result = whisper_model.align(temp_file, exact_text, language=alignment_language)
            response = _format_to_openai_verbose_json(result, engine_type)
        else:
            # Execute transcription / 执行转录
            if engine_type == "stable-ts":
                result = whisper_model.transcribe(
                    temp_file,
                    language=language,
                    initial_prompt=prompt,
                    word_timestamps=True,
                )
                response = _format_to_openai_verbose_json(result, engine_type)
            else:
                # Native faster-whisper / 原生 faster-whisper
                use_vad = vad_filter.lower() in ("true", "1", "yes", "on")
                transcribe_kwargs = dict(
                    initial_prompt=prompt,
                    language=language,
                    word_timestamps=True,
                    beam_size=1,
                    temperature=temperature,
                )
                if use_vad:
                    transcribe_kwargs["vad_filter"] = True
                    transcribe_kwargs["vad_parameters"] = dict(
                        min_silence_duration_ms=vad_min_silence_duration_ms,
                        speech_pad_ms=vad_speech_pad_ms,
                        threshold=vad_threshold,
                    )
                    logger.info(
                        f"VAD enabled / VAD 已启用: "
                        f"min_silence={vad_min_silence_duration_ms}ms, "
                        f"speech_pad={vad_speech_pad_ms}ms, "
                        f"threshold={vad_threshold}"
                    )
                segments, info = whisper_model.transcribe(temp_file, **transcribe_kwargs)
                response = _format_to_openai_verbose_json((segments, info), engine_type)

        # Return based on requested format / 根据请求格式返回
        if response_format == "verbose_json":
            return JSONResponse(content=response)
        elif response_format == "json":
            return JSONResponse(content={"text": response["text"]})
        elif response_format == "text":
            from fastapi.responses import PlainTextResponse
            return PlainTextResponse(content=response["text"])
        elif response_format == "srt":
            return _generate_srt_from_response(response)
        elif response_format == "vtt":
            return _generate_vtt_from_response(response)
        else:
            return JSONResponse(content={"text": response["text"]})

    except HTTPException:
        raise
    except Exception as e:
        # Catch ffmpeg/av decoding errors and return 400 instead of 500
        # 捕获 ffmpeg/av 解码错误并返回 400 而非 500
        error_type = type(e).__name__
        error_msg = str(e)
        logger.warning(f"Audio decode error / 音频解码错误: {error_type}: {error_msg}")

        # Check for common decode errors / 检查常见解码错误
        if "EOFError" in error_type or "End of file" in error_msg:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": "The audio file is incomplete or corrupted. Please upload a valid audio file.",
                        "type": "invalid_request_error"
                    }
                }
            )
        elif "Invalid data" in error_msg or "decoding" in error_msg.lower():
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": f"Unable to decode the audio file: {error_msg}",
                        "type": "invalid_request_error"
                    }
                }
            )
        else:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": {
                        "message": f"Internal server error during transcription: {error_msg}",
                        "type": "server_error"
                    }
                }
            )

    finally:
        # Cleanup temp file / 清理临时文件
        if os.path.exists(temp_file):
            os.remove(temp_file)


def _format_to_openai_verbose_json(result_obj, engine_type: str):
    """Unified output adapter: converts different engine outputs to standard OpenAI format
    统一输出适配器：将不同引擎的输出转换为标准 OpenAI 格式
    """
    response = {
        "task": "transcribe",
        "language": "unknown",
        "duration": 0.0,
        "text": "",
        "words": [],
        "segments": []
    }

    if engine_type == "stable-ts":
        # stable-ts returns a result object with segments / stable-ts 返回包含 segments 的结果对象
        response["language"] = getattr(result_obj, 'language', 'unknown')
        response["duration"] = getattr(result_obj, 'duration', 0.0)
        response["text"] = getattr(result_obj, 'text', '').strip()

        # Process segments / 处理 segments
        for seg in result_obj.segments if hasattr(result_obj, 'segments') else []:
            seg_dict = {
                "id": getattr(seg, 'id', 0),
                "seek": getattr(seg, 'seek', 0),
                "start": round(getattr(seg, 'start', 0.0), 3),
                "end": round(getattr(seg, 'end', 0.0), 3),
                "text": getattr(seg, 'text', '').strip(),
                "tokens": getattr(seg, 'tokens', []),
                "temperature": 0.0,
                "avg_logprob": 0.0,
                "compression_ratio": 1.0,
                "no_speech_prob": 0.0,
            }
            response["segments"].append(seg_dict)

            # Update duration / 更新时长
            if seg_dict["end"] > response["duration"]:
                response["duration"] = seg_dict["end"]

            # Extract word-level timestamps / 提取词级时间戳
            if hasattr(seg, 'words') and seg.words:
                for w in seg.words:
                    response["words"].append({
                        "word": getattr(w, 'word', getattr(w, 'text', '')).strip(),
                        "start": round(getattr(w, 'start', 0.0), 3),
                        "end": round(getattr(w, 'end', 0.0), 3)
                    })
    else:
        # faster-whisper returns (segments_generator, info_obj)
        # faster-whisper 返回 (segments_generator, info_obj)
        segments_gen, info = result_obj
        response["language"] = getattr(info, 'language', 'unknown')
        response["duration"] = round(getattr(info, 'duration', 0.0), 2)

        for seg_id, seg in enumerate(segments_gen):
            seg_dict = {
                "id": seg_id,
                "seek": seg_id,
                "start": round(seg.start, 3),
                "end": round(seg.end, 3),
                "text": seg.text.strip(),
                "tokens": [],
                "temperature": 0.0,
                "avg_logprob": 0.0,
                "compression_ratio": 1.0,
                "no_speech_prob": 0.0,
            }
            response["segments"].append(seg_dict)
            response["text"] += seg.text

            # Extract word-level timestamps / 提取词级时间戳
            if hasattr(seg, 'words') and seg.words:
                for w in seg.words:
                    response["words"].append({
                        "word": w.word.strip(),
                        "start": round(w.start, 3),
                        "end": round(w.end, 3)
                    })

    return response


def _generate_srt(segments):
    """Generate SRT subtitle format / 生成 SRT 字幕格式"""
    srt_content = ""
    for i, segment in enumerate(segments, 1):
        srt_content += f"{i}\n"
        srt_content += f"{_format_timestamp(segment.start)} --> {_format_timestamp(segment.end)}\n"
        srt_content += f"{segment.text.strip()}\n\n"
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse(content=srt_content)


def _generate_srt_from_response(response: dict):
    """Generate SRT from formatted response / 从格式化响应生成 SRT"""
    srt_content = ""
    for i, seg in enumerate(response["segments"], 1):
        srt_content += f"{i}\n"
        srt_content += f"{_format_timestamp(seg['start'])} --> {_format_timestamp(seg['end'])}\n"
        srt_content += f"{seg['text']}\n\n"
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse(content=srt_content)


def _generate_vtt_from_response(response: dict):
    """Generate VTT from formatted response / 从格式化响应生成 VTT"""
    vtt_content = "WEBVTT\n\n"
    for seg in response["segments"]:
        vtt_content += f"{_format_timestamp(seg['start'], vtt=True)} --> {_format_timestamp(seg['end'], vtt=True)}\n"
        vtt_content += f"{seg['text']}\n\n"
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse(content=vtt_content)


def _generate_vtt(segments):
    """Generate VTT subtitle format / 生成 VTT 字幕格式"""
    vtt_content = "WEBVTT\n\n"
    for i, segment in enumerate(segments, 1):
        vtt_content += f"{_format_timestamp(segment.start, vtt=True)} --> {_format_timestamp(segment.end, vtt=True)}\n"
        vtt_content += f"{segment.text.strip()}\n\n"
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse(content=vtt_content)


def _format_timestamp(seconds: float, vtt: bool = False):
    """Format timestamp / 格式化时间戳"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    if vtt:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
