import asyncio
import hashlib
import json
import re
import time
from typing import Any, Optional, Dict, Tuple, List

import httpx
from core.plugin import BasePlugin, logger, register_tool as tool, on, Priority
from core.chat.message_utils import KiraMessageEvent, KiraMessageBatchEvent
from core.chat.message_elements import Text, At
from core.chat import MessageChain


def _protect_segments(text: str, protect_idents: bool = False) -> Tuple[str, List[str]]:
    """保护代码块 / 行内代码 / URL（可选：owner/repo 与含下划线标识符）。"""
    buckets: List[str] = []

    def _save(m):
        buckets.append(m.group(0))
        return f"\x00KEEP{len(buckets)-1}\x00"

    text = re.sub(r"```[\s\S]*?```", _save, text)
    text = re.sub(r"`[^`\n]+`", _save, text)
    text = re.sub(r"https?://[^\s<>\"']+", _save, text)
    if protect_idents:
        # owner/repo、路径、snake_case 等：禁止被斜体/粗体规则误伤
        text = re.sub(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_./-]+", _save, text)
        text = re.sub(r"\b[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+\b", _save, text)
    return text, buckets


def _restore_segments(text: str, buckets: List[str]) -> str:
    for i, raw in enumerate(buckets):
        text = text.replace(f"\x00KEEP{i}\x00", raw)
    return text


# Unicode 数学字母（仅映射拉丁/数字；中文等原样保留）
_BOLD_MAP = str.maketrans(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
    "𝗔𝗕𝗖𝗗𝗘𝗙𝗚𝗛𝗜𝗝𝗞𝗟𝗠𝗡𝗢𝗣𝗤𝗥𝗦𝗧𝗨𝗩𝗪𝗫𝗬𝗭𝗮𝗯𝗰𝗱𝗲𝗳𝗴𝗵𝗶𝗷𝗸𝗹𝗺𝗻𝗼𝗽𝗾𝗿𝘀𝘁𝘂𝘃𝘄𝘅𝘆𝘇𝟬𝟭𝟮𝟯𝟰𝟱𝟲𝟳𝟴𝟵",
)
_ITALIC_MAP = str.maketrans(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
    "𝘈𝘉𝘊𝘋𝘌𝘍𝘎𝘏𝘐𝘑𝘒𝘓𝘔𝘕𝘖𝘗𝘘𝘙𝘚𝘛𝘜𝘝𝘞𝘟𝘠𝘡𝘢𝘣𝘤𝘥𝘦𝘧𝘨𝘩𝘪𝘫𝘬𝘭𝘮𝘯𝘰𝘱𝘲𝘳𝘴𝘵𝘶𝘷𝘸𝘹𝘺𝘻",
)
_MONO_MAP = str.maketrans(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
    "𝙰𝙱𝙲𝙳𝙴𝙵𝙶𝙷𝙸𝙹𝙺𝙻𝙼𝙽𝙾𝙿𝚀𝚁𝚂𝚃𝚄𝚅𝚆𝚇𝚈𝚉𝚊𝚋𝚌𝚍𝚎𝚏𝚐𝚑𝚒𝚓𝚔𝚕𝚖𝚗𝚘𝚙𝚚𝚛𝚜𝚝𝚞𝚟𝚠𝚡𝚢𝚣𝟶𝟷𝟸𝟹𝟺𝟻𝟼𝟽𝟾𝟿",
)


def _to_bold(s: str) -> str:
    return s.translate(_BOLD_MAP)


def _to_italic(s: str) -> str:
    return s.translate(_ITALIC_MAP)


def _to_monospace(s: str) -> str:
    return s.translate(_MONO_MAP)


def sanitize_qq_text(text: str) -> str:
    """仅剥离 Markdown 语法标记，绝不改写实际字符（尤其是下划线 _）。"""
    if not text:
        return text

    text, buckets = _protect_segments(text, protect_idents=False)

    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text, flags=re.DOTALL)
    # 不处理 __bold__，避免误伤下划线
    text = re.sub(r"(?<!\*)\*(?!\*)([^*\n]+?)(?<!\*)\*(?!\*)", r"\1", text)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^>\s?", "", text, flags=re.MULTILINE)

    text = _restore_segments(text, buckets)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _has_cjk(s: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]", s))


def _to_fullwidth_alnum(s: str) -> str:
    """拉丁字母/数字 → 全角（BMP，QQ/NapCat 兼容，比数学粗体更稳）。"""
    out = []
    for ch in s:
        o = ord(ch)
        if 0x41 <= o <= 0x5A:
            out.append(chr(0xFF21 + (o - 0x41)))
        elif 0x61 <= o <= 0x7A:
            out.append(chr(0xFF41 + (o - 0x61)))
        elif 0x30 <= o <= 0x39:
            out.append(chr(0xFF10 + (o - 0x30)))
        else:
            out.append(ch)
    return "".join(out)


def _style_bold(s: str) -> str:
    """粗体：英文全角 + 中文【】，替代 Markdown **。"""
    s = s.strip()
    if not s:
        return s
    if _has_cjk(s):
        return f"【{_to_fullwidth_alnum(s)}】"
    return _to_fullwidth_alnum(s)


def _style_italic(s: str) -> str:
    """斜体：统一「」，替代 Markdown *。"""
    s = s.strip()
    if not s:
        return s
    return f"「{s}」"


def stylize_qq_text(text: str) -> str:
    """Markdown 强调 → 美观全角/【】/「」（不用数学字母，避免 NapCat 伪造转发失败）。"""
    if not text:
        return text

    text, buckets = _protect_segments(text, protect_idents=True)

    text = re.sub(
        r"^#{1,6}\s*(.+?)\s*$",
        lambda m: _style_bold(m.group(1)),
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r"\*\*(.+?)\*\*",
        lambda m: _style_bold(m.group(1)),
        text,
        flags=re.DOTALL,
    )
    text = re.sub(
        r"(?<!\*)\*(?!\*)([^*\n]+?)(?<!\*)\*(?!\*)",
        lambda m: _style_italic(m.group(1)),
        text,
    )

    text = _restore_segments(text, buckets)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_empty_template(tmpl: Any) -> bool:
    """空字符串 / 空对象 / 空 lines → 视为未配置。"""
    if tmpl is None:
        return True
    if isinstance(tmpl, str):
        return not tmpl.strip()
    if isinstance(tmpl, dict):
        if not tmpl:
            return True
        if "lines" in tmpl:
            lines = tmpl.get("lines")
            return not lines
        for k in ("content", "text", "value", "body"):
            if k in tmpl:
                return not str(tmpl.get(k) or "").strip()
        return False
    if isinstance(tmpl, (list, tuple)):
        return len(tmpl) == 0
    return False


# ============================================================
# DeepWiki MCP 客户端（复用原有逻辑，稍作增强）
# ============================================================

class _RetryableError(Exception):
    """标记可重试的瞬时故障（HTTP 429/5xx、网络抖动、服务端过载错误文本）。"""
    pass


# 瞬时 HTTP 状态码：重试有意义
_TRANSIENT_HTTP_CODES = {429, 500, 502, 503, 504}


# 服务端错误文本中的瞬时故障特征（如 "Server error '503 Service Temporarily Unavailable'..."）
_TRANSIENT_TEXT_RE = re.compile(
    r"(?<!\d)(?:50[0-4]|429)(?!\d)"
    r"|temporarily unavailable|service unavailable|server error"
    r"|bad gateway|overloaded|timed?\s*out|connection (?:reset|refused|closed)"
    r"|error processing question",
    re.IGNORECASE,
)


def _is_transient_text(text: str) -> bool:
    """判断服务端返回文本是否属于瞬时故障（适合重试）。"""
    return bool(_TRANSIENT_TEXT_RE.search(text or ""))


class DeepWikiClient:
    def __init__(self, mcp_url: str, protocol_version: str, timeout: float, max_retries: int = 3):
        self.mcp_url = mcp_url
        self.protocol_version = protocol_version
        self.timeout = timeout
        # 瞬时故障（429/5xx/网络抖动/服务端过载）最大重试次数；退避 1s/2s/4s...
        self.max_retries = max_retries
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def _call_mcp_tool(self, tool_name: str, arguments: dict) -> str:
        """通用 MCP 工具调用，支持 ask_question / read_wiki_structure / read_wiki_contents

        对瞬时故障自动指数退避重试（最多 max_retries 次重试）：
        HTTP 429/5xx、网络异常/超时、服务端错误文本（如 "Server error '503...'"）。
        全部重试仍失败才返回错误串，上层（LLM 工具 / /dw 命令）完全无感。
        """
        client = await self._get_client()
        headers = {
            "Content-Type": "application/json",
            "MCP-Protocol-Version": self.protocol_version,
            "Accept": "application/json, text/event-stream",
        }
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments,
            },
        }
        last_err = "unknown error"
        for attempt in range(self.max_retries + 1):
            try:
                resp = await client.post(self.mcp_url, json=payload, headers=headers)
                # 瞬时 HTTP 状态码：直接重试
                if resp.status_code in _TRANSIENT_HTTP_CODES:
                    raise _RetryableError(f"HTTP {resp.status_code}")
                text = resp.text
                if not text:
                    raise _RetryableError("Empty response from DeepWiki MCP")

                full_answer = []
                for line in text.splitlines():
                    line = line.strip()
                    if line.startswith("data: "):
                        json_str = line[6:]
                        try:
                            data = json.loads(json_str)
                            if "result" in data and "content" in data["result"]:
                                content = data["result"]["content"]
                                if isinstance(content, list) and len(content) > 0:
                                    full_answer.append(content[0].get("text", ""))
                                elif isinstance(content, dict):
                                    full_answer.append(content.get("text", ""))
                            elif "error" in data:
                                msg = f"MCP error: {data['error'].get('message', 'Unknown')}"
                                if _is_transient_text(msg):
                                    raise _RetryableError(msg)
                                return msg
                        except json.JSONDecodeError:
                            continue

                if full_answer:
                    answer = "\n".join(full_answer)
                    # 服务端把上游错误当正常文本返回（如 "Error processing question: Server error '503...'"）
                    if _is_transient_text(answer):
                        raise _RetryableError(answer)
                    return answer

                # fallback: 直接 JSON
                data = json.loads(text)
                if "result" in data and "content" in data["result"]:
                    content = data["result"]["content"]
                    if isinstance(content, list) and len(content) > 0:
                        answer = content[0].get("text", "")
                        if _is_transient_text(answer):
                            raise _RetryableError(answer)
                        return answer
                    elif isinstance(content, dict):
                        answer = content.get("text", "")
                        if _is_transient_text(answer):
                            raise _RetryableError(answer)
                        return answer
                elif "error" in data:
                    msg = f"MCP error: {data['error'].get('message', 'Unknown')}"
                    if _is_transient_text(msg):
                        raise _RetryableError(msg)
                    return msg
                else:
                    return f"Unexpected response: {str(data)[:200]}"

            except _RetryableError as e:
                last_err = str(e)
                if attempt >= self.max_retries:
                    break
                logger.warning(
                    f"DeepWiki MCP transient error (attempt {attempt + 1}/{self.max_retries + 1}): {e}"
                )
                await asyncio.sleep(2 ** attempt)
            except (httpx.HTTPStatusError, httpx.TransportError, httpx.TimeoutException, asyncio.TimeoutError) as e:
                last_err = str(e)
                if attempt >= self.max_retries:
                    break
                logger.warning(
                    f"DeepWiki MCP network error (attempt {attempt + 1}/{self.max_retries + 1}): {e}"
                )
                await asyncio.sleep(2 ** attempt)
            except Exception as e:
                logger.error(f"DeepWiki MCP call failed: {e}")
                return f"DeepWiki request failed: {str(e)}"

        logger.error(f"DeepWiki MCP call failed after {self.max_retries + 1} attempts: {last_err}")
        return f"DeepWiki request failed: {last_err}"

    async def ask_question(self, repo: str, question: str) -> str:
        return await self._call_mcp_tool(
            "ask_question",
            {"repoName": repo, "question": question}
        )

    async def read_wiki_structure(self, repo: str) -> str:
        return await self._call_mcp_tool(
            "read_wiki_structure",
            {"repoName": repo}
        )

    async def read_wiki_contents(self, repo: str, topic: str = "") -> str:
        args = {"repoName": repo}
        if topic:
            args["topic"] = topic
        return await self._call_mcp_tool("read_wiki_contents", args)

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None


# ============================================================
# GitHub 多路搜索
# ============================================================

async def _github_search(
    keyword: str,
    github_token: str,
    per_page: int = 10,
    sort: Optional[str] = None,
    order: str = "desc"
) -> List[dict]:
    """单次 GitHub 仓库搜索，返回原始 items 列表"""
    headers = {}
    if github_token:
        headers["Authorization"] = f"token {github_token}"

    params = {
        "q": keyword,
        "per_page": per_page,
    }
    if sort:
        params["sort"] = sort
        params["order"] = order

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                "https://api.github.com/search/repositories",
                params=params,
                headers=headers,
                timeout=10.0
            )
            resp.raise_for_status()
            data = resp.json()
            items = data.get("items", []) if isinstance(data, dict) else []
            return [i for i in items if isinstance(i, dict)]
    except Exception as e:
        logger.error(f"GitHub search failed for '{keyword}': {e}")
        return []


async def _multi_path_search_repositories(
    keyword: str,
    github_token: str,
    max_results: int = 5
) -> List[dict]:
    """
    多路查询融合：
    1. 精确名称匹配
    2. 描述/话题匹配
    3. 通用搜索（无限定）
    合并去重，按简单得分排序
    """
    tasks = [
        _github_search(f"{keyword} in:name", github_token, per_page=max_results * 2),
        _github_search(f"{keyword} in:description,topics", github_token, per_page=max_results * 2),
        _github_search(keyword, github_token, per_page=max_results * 2),
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    merged: Dict[str, dict] = {}
    for path_idx, items in enumerate(results):
        if isinstance(items, Exception) or not items:
            continue
        for item in items:
            full_name = item.get("full_name")
            if not full_name or full_name in merged:
                continue

            score = 0
            name = (item.get("name") or "").lower()
            desc = (item.get("description") or "").lower()
            k_lower = keyword.lower()

            # 名称精确包含
            if k_lower in name:
                score += 3
            # 描述/话题包含
            if k_lower in desc:
                score += 2

            # stars 加权（log 近似）
            stars = item.get("stargazers_count", 0) or 0
            score += min(stars // 100, 5)

            # 最近更新时间（简单加分）
            pushed = item.get("pushed_at") or ""
            if pushed.startswith("2026") or pushed.startswith("2025"):
                score += 1

            merged[full_name] = {
                "full_name": full_name,
                "stars": stars,
                "description": item.get("description", ""),
                "score": score,
                "html_url": item.get("html_url", ""),
            }

    sorted_results = sorted(
        merged.values(),
        key=lambda x: (x["score"], x["stars"]),
        reverse=True
    )
    return sorted_results[:max_results]


def _format_repo_candidates(repos: List[dict], cmd_prefix: str = "/dw", extra_hint: str = "") -> str:
    if not repos:
        return "未找到相关仓库。可换个关键词再试，或直接发送 owner/repo。"

    p = (cmd_prefix or "/dw").strip() or "/dw"
    lines = ["搜索到以下候选仓库（按相关性排序）：", ""]
    for idx, r in enumerate(repos, 1):
        desc = r.get("description", "") or ""
        # 保留原文下划线等字符，仅截断长度
        if len(desc) > 100:
            desc = desc[:100] + "..."
        full = r.get("full_name") or ""
        stars = r.get("stars", 0)
        lines.append(f"{idx}. {full}  ⭐{stars}")
        if desc:
            lines.append(f"   {desc}")
        lines.append(f"   选用：{p} {idx}   或   {p} {full}")
        lines.append("")

    lines.append("——怎么选——")
    lines.append(f"• {p} 1          → 从候选列表查询第 1 个仓库")
    lines.append(f"• {p} 2          → 查询第 2 个…")
    lines.append(f"• {p} owner/repo → 直接指定仓库")
    lines.append(f"• {p} <问题>     → 选定后继续追问（需已有上下文）")
    lines.append(f"• {p} ?          → 查看当前上下文仓库")
    lines.append(f"• {p} clear      → 清除上下文，重新查询仓库来提问")
    if extra_hint and str(extra_hint).strip() and str(extra_hint).strip() not in ("{}", "[]"):
        lines.append(str(extra_hint).rstrip())
    return "\n".join(lines).rstrip()


# ============================================================
# 主插件
# ============================================================

class DeepWikiPlugin(BasePlugin):
    def __init__(self, ctx, cfg: dict):
        super().__init__(ctx, cfg)

        def _sec(name: str) -> dict:
            v = cfg.get(name)
            return v if isinstance(v, dict) else {}

        # 旧版单一大分组里各字段的 schema 默认值（用于判断新分组的值是否「只是默认值」）
        _CMD_LEGACY_DEFAULTS = {
            "enable_command": False,
            "command_words": ["/dw", "dw:", "/deepwiki"],
            "default_question": "请总结这个仓库的核心功能、主要架构以及如何使用或贡献。",
            "isolate_context_by_user": False,
            "default_repo_list": [],
            "reset_keeps_preset_repo": True,
            "llm_bind_preset_repo": False,
            "clear_command_words": ["clear", "重置", "清空", "reset", "清除", "清除上下文", "重置上下文"],
            "status_command_words": ["?", "？", "status", "ctx", "context", "当前", "当前仓库"],
        }
        _FWD_LEGACY_DEFAULTS = {
            "enable_auto_forward": True,
            "qq_rich_text_mode": "sanitize",
            "force_forward_all": True,
            "use_length_threshold": False,
            "forward_threshold": 400,
            "send_intro_message": False,
            "intro_template": {"lines": ["DeepWiki 查询结果", "仓库：{repo}", "问题：{question}"]},
            "prepend_metadata_in_card": True,
            "forward_metadata_template": {"lines": ["【DeepWiki 查询】", "仓库：{repo}", "问题：{question}", ""]},
            "enable_private_forward_fallback": True,
            "forward_node_max_chars": 800,
            "forward_api_timeout": 60,
            "enable_forward_plain_fallback": False,
            "append_operation_guide": True,
            "operation_guide_text": {"lines": []},
        }

        def _sec_merge_smart(old_names, new_names, defaults: dict, group_label: str) -> dict:
            """分组拆分后的兼容读取。

            KiraAI 的 _ensure_plugin_config 会按新 schema 给新分组补默认值并保留旧分组，
            WebUI 增量保存也不会清除旧分组 —— 若简单让新分组覆盖旧分组，
            新分组的「默认值」会顶掉用户在旧分组的真实配置。
            规则：逐键判断，新分组的值等于 schema 默认值（未被用户改过）时，
            回退使用旧分组的用户值。
            """
            new_merged: dict = {}
            for n in new_names:
                v = cfg.get(n)
                if isinstance(v, dict):
                    new_merged.update(v)
            old_merged: dict = {}
            for n in old_names:
                v = cfg.get(n)
                if isinstance(v, dict):
                    old_merged.update(v)
            out = dict(new_merged)
            legacy_used = []
            for k, ov in old_merged.items():
                if k not in out:
                    out[k] = ov
                    legacy_used.append(k)
                elif k in defaults and out.get(k) == defaults[k] and ov != defaults[k]:
                    out[k] = ov
                    legacy_used.append(k)
            if legacy_used:
                logger.warning(
                    f"DeepWiki: 以下配置项仍读取自旧版分组（{group_label}）：{legacy_used}。"
                    "如需以新界面显示的值为准，请在 WebUI 插件配置页直接重新保存一次。"
                )
            return out

        conn = _sec("section_connection")
        storage = _sec("section_storage")
        search = _sec("section_search")
        # 兼容旧版单一大分组：section_command / section_forward
        cmd_section = _sec_merge_smart(
            ("section_command",),
            ("section_command_basic", "section_command_context"),
            _CMD_LEGACY_DEFAULTS,
            "section_command",
        )
        tool_section = _sec("section_tool")
        fwd_section = _sec_merge_smart(
            ("section_forward",),
            ("section_forward_basic", "section_forward_card", "section_forward_compat"),
            _FWD_LEGACY_DEFAULTS,
            "section_forward",
        )

        # --- 基础（兼容扁平旧配置） ---
        self.enabled = bool(conn.get("enabled", cfg.get("enabled", True)))
        self.mcp_url = conn.get("mcp_url") or cfg.get("mcp_url") or "https://mcp.deepwiki.com/mcp"
        self.protocol_version = conn.get("protocol_version") or cfg.get("protocol_version") or "2025-11-25"
        self.timeout = float(conn.get("timeout", cfg.get("timeout", 60)) or 60)
        self.github_token = conn.get("github_token") or cfg.get("github_token") or ""

        # --- 缓存 ---
        self.cache_ttl = int(storage.get("cache_ttl", cfg.get("cache_ttl", 3600)) or 0)
        self.auto_save = bool(storage.get("auto_save_to_kb", cfg.get("auto_save_to_kb", False)))
        self.target_kb = storage.get("target_kb_id") or cfg.get("target_kb_id") or "deepwiki_cache"

        # --- 搜索 ---
        self.max_search_results = int(search.get("max_search_results", cfg.get("max_search_results", 5)) or 5)
        self.use_multi_path = bool(search.get("use_multi_path_search", cfg.get("use_multi_path_search", True)))

        # --- 命令 ---
        self.enable_command = bool(cmd_section.get("enable_command", False))
        self.command_words = list(cmd_section.get("command_words") or ["/dw", "dw:", "/deepwiki"])
        # 按长度降序，避免短命令抢先匹配
        self.command_words = sorted([str(w) for w in self.command_words if w], key=len, reverse=True)
        self.default_question = cmd_section.get(
            "default_question",
            "请总结这个仓库的核心功能、主要架构以及如何使用或贡献。",
        )
        self.isolate_context_by_user = bool(cmd_section.get("isolate_context_by_user", False))

        self.clear_command_words = list(cmd_section.get("clear_command_words") or [
            "clear", "重置", "清空", "reset", "清除", "清除上下文", "重置上下文"
        ])
        self.status_command_words = list(cmd_section.get("status_command_words") or [
            "?", "？", "status", "ctx", "context", "当前", "当前仓库"
        ])

        raw_presets: List[str] = cmd_section.get("default_repo_list", []) or []
        self.default_repo_presets: List[Tuple[str, str]] = []
        for line in raw_presets:
            if isinstance(line, str) and ";" in line:
                k, v = [x.strip() for x in line.split(";", 1)]
                if k and v:
                    self.default_repo_presets.append((k, v))

        # 重置命令是否保留预设仓库（默认开：预设了仓库的会话/用户不会被清回搜索阶段）
        self.reset_keeps_preset_repo = bool(cmd_section.get("reset_keeps_preset_repo", True))
        # LLM 自然语言调用是否默认绑定预设/当前仓库（默认关）
        self.llm_bind_preset_repo = bool(cmd_section.get("llm_bind_preset_repo", False))

        # --- LLM 工具 ---
        self.enable_llm_tool = bool(tool_section.get("enable_llm_tool", True))

        # --- 转发 ---
        self.enable_auto_forward = bool(fwd_section.get("enable_auto_forward", True))
        self.force_forward_all = bool(fwd_section.get("force_forward_all", True))
        self.use_length_threshold = bool(fwd_section.get("use_length_threshold", False))
        self.forward_threshold = int(fwd_section.get("forward_threshold", 400) or 400)
        self.send_intro_message = bool(fwd_section.get("send_intro_message", False))

        self.intro_template = fwd_section.get("intro_template", {
            "lines": [
                "DeepWiki 查询结果",
                "仓库：{repo}",
                "问题：{question}",
            ]
        })
        self.prepend_metadata_in_card = bool(fwd_section.get("prepend_metadata_in_card", True))
        self.forward_metadata_template = fwd_section.get("forward_metadata_template", {
            "lines": [
                "【DeepWiki 查询】",
                "仓库：{repo}",
                "问题：{question}",
                "",
            ]
        })
        self.enable_private_forward_fallback = bool(fwd_section.get("enable_private_forward_fallback", True))
        # 合并转发失败后是否把全文拆成多条普通消息发送（默认关：只提示原因）
        self.enable_forward_plain_fallback = bool(
            fwd_section.get("enable_forward_plain_fallback", False)
        )
        self.forward_node_max_chars = int(
            fwd_section.get("forward_node_max_chars", 800) or 800
        )
        self.forward_node_max_chars = max(400, min(self.forward_node_max_chars, 3500))
        # NapCat send_action 默认 timeout=10s，合并转发常需更久
        self.forward_api_timeout = float(
            fwd_section.get("forward_api_timeout", 60) or 60
        )
        self.forward_api_timeout = max(15.0, min(self.forward_api_timeout, 180.0))
        self.qq_rich_text_mode = fwd_section.get("qq_rich_text_mode") or "sanitize"
        self.append_operation_guide = bool(fwd_section.get("append_operation_guide", True))
        self.operation_guide_text = fwd_section.get("operation_guide_text") or {"lines": []}
        # 候选提示：优先 section_search，兼容旧配置写在 section_forward 的情况
        self.candidate_hint_text = (
            search.get("candidate_hint_text")
            if search.get("candidate_hint_text") is not None
            else fwd_section.get("candidate_hint_text")
        ) or {"lines": []}

        # 运行时状态（必须初始化）
        self._client: Optional[DeepWikiClient] = None
        self._answer_cache: Dict[str, Tuple[str, float]] = {}
        self._search_cache: Dict[str, Tuple[List[dict], float]] = {}
        self._last_repo: Dict[str, str] = {}
        self._last_candidates: Dict[str, List[dict]] = {}

    def _render_template(self, tmpl: Any, **kwargs) -> str:
        """将配置中的模板（支持旧字符串或 JSON 对象）渲染为最终文本。

        空对象 {} / 空字符串 → 返回空串（绝不输出字面量 "{}"）。
        JSON 支持：
          { "lines": ["line1", "line2", ...] } → 用 \\n 拼接
          { "content": "..." } 或 { "text": "..." }
        """
        if is_empty_template(tmpl):
            return ""

        if isinstance(tmpl, str):
            try:
                return tmpl.format(**kwargs) if kwargs else tmpl
            except Exception:
                return tmpl

        if isinstance(tmpl, dict):
            if "lines" in tmpl and isinstance(tmpl["lines"], (list, tuple)):
                raw = "\n".join(str(x) for x in tmpl["lines"])
                try:
                    return raw.format(**kwargs) if kwargs else raw
                except Exception:
                    return raw
            for k in ("content", "text", "value", "body"):
                if k in tmpl:
                    raw = str(tmpl[k])
                    try:
                        return raw.format(**kwargs) if kwargs else raw
                    except Exception:
                        return raw
            # 有内容但无标准字段：不 str(dict)，避免 "{}"
            return ""

        if isinstance(tmpl, (list, tuple)):
            raw = "\n".join(str(x) for x in tmpl)
            try:
                return raw.format(**kwargs) if kwargs else raw
            except Exception:
                return raw

        s = str(tmpl)
        if s.strip() in ("{}", "[]", "None"):
            return ""
        try:
            return s.format(**kwargs) if kwargs else s
        except Exception:
            return s

    async def initialize(self):
        if not self.enabled:
            logger.info("DeepWiki plugin disabled")
            return
        self._client = DeepWikiClient(self.mcp_url, self.protocol_version, self.timeout)
        logger.info(
            "DeepWiki plugin initialized: "
            f"enable_command={self.enable_command}, "
            f"presets={self.default_repo_presets}, "
            f"reset_keeps_preset_repo={self.reset_keeps_preset_repo}, "
            f"llm_bind_preset_repo={self.llm_bind_preset_repo}, "
            f"isolate_by_user={self.isolate_context_by_user}, "
            f"force_forward_all={self.force_forward_all}"
        )

    async def terminate(self):
        if self._client:
            await self._client.close()
            self._client = None
        self._answer_cache.clear()
        self._search_cache.clear()
        logger.info("DeepWiki plugin terminated")

    # ============================================================
    # 缓存辅助
    # ============================================================

    def _cache_key(self, repo: str, question: str) -> str:
        raw = f"{repo}|{question}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _get_cached_answer(self, key: str) -> Optional[str]:
        if self.cache_ttl <= 0:
            return None
        entry = self._answer_cache.get(key)
        if entry and (time.time() - entry[1] < self.cache_ttl):
            return entry[0]
        return None

    def _set_cached_answer(self, key: str, answer: str):
        if self.cache_ttl > 0:
            self._answer_cache[key] = (answer, time.time())

    def _get_cached_search(self, keyword: str) -> Optional[List[dict]]:
        entry = self._search_cache.get(keyword)
        if entry and (time.time() - entry[1] < 60):  # 搜索缓存 60 秒
            return entry[0]
        return None

    def _set_cached_search(self, keyword: str, repos: List[dict]):
        self._search_cache[keyword] = (repos, time.time())

    def _sanitize(self, text: str) -> str:
        """
        根据 qq_rich_text_mode 处理输出：
        - off: 原样
        - sanitize: 仅移除 Markdown 标记，保留 _ 与代码
        - stylize: Unicode/【】富文本，保护代码与下划线
        """
        mode = getattr(self, "qq_rich_text_mode", "sanitize") or "sanitize"
        if mode == "off":
            return text
        if mode == "stylize":
            return stylize_qq_text(text)
        return sanitize_qq_text(text)

    def _primary_cmd(self) -> str:
        """展示用主命令词（优先 /dw）。"""
        for w in self.command_words:
            if w == "/dw":
                return w
        return self.command_words[0] if self.command_words else "/dw"

    def _extract_command_text(self, event: KiraMessageEvent) -> str:
        """提取纯文本命令，忽略 @ 元素；支持「@机器人 /dw ...」。"""
        parts: List[str] = []
        for elem in event.message.chain:
            if isinstance(elem, At):
                continue
            if isinstance(elem, Text):
                parts.append(elem.text or "")
        text = "".join(parts).strip()
        # 去掉残留的 @昵称 前缀（部分适配器把 @ 编进 Text）
        text = re.sub(r"^(?:@\S+\s*)+", "", text).strip()
        return text

    def _match_command(self, text: str) -> Optional[Tuple[str, str]]:
        """返回 (matched_cmd, raw_query)。"""
        if not text:
            return None
        for cmd in self.command_words:
            if not cmd:
                continue
            if text == cmd:
                return cmd, ""
            if text.startswith(cmd + " ") or text.startswith(cmd + ":"):
                return cmd, text[len(cmd):].lstrip(" :")
            # 允许命令后直接跟数字：/dw1 不支持；必须空格
        return None

    def _looks_like_new_keyword(self, text: str) -> bool:
        """单 token、不像问句 → 视为新关键词搜索（便于切换项目）。"""
        t = (text or "").strip()
        if not t or any(ch.isspace() for ch in t):
            return False
        if any(x in t for x in ("？", "?", "。", "！", "!", "，", ",")):
            return False
        if len(t) > 48:
            return False
        if re.search(r"(怎么|如何|什么|为什么|哪些|是否|能否|怎样|哪里|多少)", t):
            return False
        if "/" in t:
            return False
        return True

    def _is_repo_query_failure(self, answer: str) -> bool:
        if not answer:
            return True
        low = answer.lower()
        markers = (
            "repository not found",
            "requested repos",
            "visit https://deepwiki.com to index",
            "to index it",
            "not indexed",
            "查询失败",
            "mcp error",
            "deepwiki request failed",
            "empty response",
            # 服务端错误文本（含瞬时故障），防止被当正常回答发出/缓存
            "error processing question",
            "server error",
            "temporarily unavailable",
            "service unavailable",
            "bad gateway",
            "overloaded",
        )
        if any(m in low for m in markers):
            return True
        # HTTP 5xx / 429 状态码数字（前后非数字，避免误伤回答正文）
        if re.search(r"(?<!\d)(?:50[0-4]|429)(?!\d)", low):
            return True
        # 常见失败组合
        if "error" in low and ("failed" in low or "not found" in low):
            return True
        return False

    # ============================================================
    # 关键词命令拦截（直接触发，隔离上下文）
    # ============================================================

    @on.im_message(priority=Priority.HIGH)
    async def handle_command(self, event: KiraMessageEvent):
        """拦截 /dw 命令（含 @机器人 /dw），不走 LLM。"""
        if not self.enabled or not self.enable_command:
            return

        text = self._extract_command_text(event)
        matched = self._match_command(text)
        if not matched:
            return

        matched_cmd, raw_query = matched

        # 丢弃消息并中止后续钩子，避免进入默认 chat / LLM
        event.discard(force=True)
        event.stop()

        asyncio.create_task(self._process_dw_command_bg(event, matched_cmd, raw_query))

    async def _process_dw_command_bg(self, event: KiraMessageEvent, matched_cmd: str, raw_query: str):
        """后台处理 /dw 命令，避免阻塞主事件循环。"""
        p = self._primary_cmd()
        try:
            raw_query = (raw_query or "").strip()
            ctx_key = self._get_ctx_key(event)
            session_key = self._get_sid(event)
            # 分级预设：user 专属优先，否则会话级预设接管该会话所有用户
            preset = self._apply_default_preset_if_needed(ctx_key, session_key)
            logger.info(
                f"DeepWiki cmd: ctx={ctx_key} query={raw_query[:40]!r} "
                f"bound={self._last_repo.get(ctx_key)}"
            )

            if not raw_query:
                last = self._last_repo.get(ctx_key)
                if last:
                    is_preset = bool(preset) and last == preset
                    tag = "（预设仓库）" if is_preset else ""
                    lines = [
                        f"当前上下文仓库：{last}{tag}",
                        "",
                        f"• {p} <你的问题>   → 直接向该仓库提问",
                        f"• {p} ?            → 查看当前上下文",
                        f"• {p} clear        → 清除上下文"
                        + ("（预设仓库会保持绑定）" if is_preset and self.reset_keeps_preset_repo else ""),
                        f"• {p} owner/repo   → 改查其他仓库",
                    ]
                    if not is_preset:
                        lines.append(f"• {p} <关键词>     → 重新搜索仓库")
                    reply = "\n".join(lines)
                else:
                    reply = (
                        f"用法示例：\n"
                        f"{p} xxynet/KiraAI\n"
                        f"{p} xxynet/KiraAI 这个项目怎么安装？\n"
                        f"{p} KiraAI\n"
                        f"{p} 1          （从候选列表中查询对应序号的仓库）\n"
                        f"{p} ?          （查看当前上下文仓库）\n"
                        f"{p} clear      （清除上下文，重新查询仓库来提问）"
                    )
                await self._send_maybe_forward(event, reply)
                return

            # 清除 / 重置
            if self._is_clear_intent(raw_query):
                self._clear_ctx(ctx_key)
                # 开关开启且该会话/用户配置了预设仓库：重置后仍保持绑定，不回到搜索阶段
                if self.reset_keeps_preset_repo and preset:
                    self._last_repo[ctx_key] = preset
                    await self._send_maybe_forward(
                        event,
                        f"✅ 已清除当前上下文（对话记忆与候选列表）。\n"
                        f"本会话/用户预设了仓库，重置后仍保持绑定：{preset}\n"
                        f"可直接继续提问：{p} <你的问题>\n"
                        f"查看当前上下文：{p} ?",
                    )
                    return
                await self._send_maybe_forward(
                    event,
                    f"✅ 已清除当前上下文。\n可重新发送：{p} <关键词> 或 {p} owner/repo",
                )
                return

            # 状态查询
            if self._is_status_intent(raw_query):
                last = self._last_repo.get(ctx_key)
                cands = self._last_candidates.get(ctx_key) or []
                if last:
                    msg = (
                        f"当前上下文仓库：{last}\n"
                        f"继续追问：{p} <你的问题>\n"
                        f"清除上下文：{p} clear\n"
                        f"换项目：{p} clear 后，再 {p} <新关键词>"
                    )
                else:
                    msg = (
                        "当前还没有上下文仓库。\n"
                        f"请先：{p} <关键词> 或 {p} owner/repo"
                    )
                if cands:
                    msg += f"\n最近候选数：{len(cands)}（可用 {p} 1 选择）"
                await self._send_maybe_forward(event, msg)
                return

            # 数字选择候选
            if raw_query.isdigit():
                candidates = self._last_candidates.get(ctx_key, [])
                idx = int(raw_query) - 1
                if 0 <= idx < len(candidates):
                    repo = candidates[idx]["full_name"]
                    self._last_repo[ctx_key] = repo
                    await self._execute_direct_query(event, repo, self.default_question, ctx_key=ctx_key)
                    return
                reply = (
                    f"序号无效或没有候选列表。\n"
                    f"请先发送：{p} <关键词>\n"
                    f"再发送：{p} 1"
                )
                await self._send_maybe_forward(event, reply)
                return

            parts = raw_query.split(maxsplit=1)
            first = parts[0]
            question = parts[1] if len(parts) > 1 else ""

            # owner/repo
            if "/" in first and re.match(r"^[\w.-]+/[\w.-]+$", first):
                repo = first
                self._last_repo[ctx_key] = repo
                await self._execute_direct_query(
                    event, repo, question or self.default_question, ctx_key=ctx_key
                )
                return

            last_repo = self._last_repo.get(ctx_key)
            # 有上下文：默认当追问；若像新关键词则重新搜索（便于切换项目）。
            # 例外：当前绑定的是预设仓库且「重置保留预设」开启时，会话视为锁定在该仓库，
            # 单 token 也按问题处理（切换仓库需显式 /dw owner/repo）。
            if last_repo:
                pinned = bool(preset) and self.reset_keeps_preset_repo and last_repo == preset
                if (not question) and not pinned and self._looks_like_new_keyword(raw_query):
                    # 切换关键词搜索，不沿用旧仓库
                    self._last_repo.pop(ctx_key, None)
                    await self._handle_keyword_search_and_reply(event, raw_query)
                    return
                await self._execute_direct_query(
                    event, last_repo, raw_query or self.default_question, ctx_key=ctx_key
                )
                return

            # 无上下文 → 关键词搜索
            await self._handle_keyword_search_and_reply(event, first if not question else raw_query)

        except Exception as e:
            logger.exception(f"DeepWiki 后台命令处理异常: {e}")
            try:
                await self._send_maybe_forward(event, f"处理命令时发生错误：{str(e)[:200]}")
            except Exception:
                pass

    async def _handle_keyword_search_and_reply(self, event: KiraMessageEvent, keyword: str):
        ctx_key = self._get_ctx_key(event)
        keyword = (keyword or "").strip()
        cached = self._get_cached_search(keyword)
        if cached is None:
            cached = await _multi_path_search_repositories(
                keyword, self.github_token, self.max_search_results
            )
            self._set_cached_search(keyword, cached)

        self._last_candidates[ctx_key] = cached
        # 重要：搜索阶段不自动绑定第一个仓库，避免误把后续关键词当追问

        extra = ""
        if not is_empty_template(self.candidate_hint_text):
            extra = self._render_template(
                self.candidate_hint_text, cmd=self._primary_cmd(), keyword=keyword
            )

        reply = _format_repo_candidates(
            cached, cmd_prefix=self._primary_cmd(), extra_hint=extra or ""
        )
        await self._send_maybe_forward(event, reply)

    # QQ/NapCat：单节点过长时 send_forward_msg 易超时；单条普通消息也有长度上限
    _PLAIN_MSG_MAX_CHARS = 3500

    async def _send_maybe_forward(self, event: KiraMessageEvent, text: str):
        """根据转发配置决定是普通消息还是合并转发（对候选、错误等也生效）"""
        if not self.enable_auto_forward:
            await self._send_text_reply(event, text)
            return

        # force_forward_all 开启时，全部走转发
        if self.force_forward_all:
            await self._send_as_forward_content(event, text)
            return

        # 否则只有在启用长度判断 且 超过阈值 时才转发
        if self.use_length_threshold and len(text) > self.forward_threshold:
            await self._send_as_forward_content(event, text)
        else:
            await self._send_text_reply(event, text)

    @staticmethod
    def _split_text_chunks(text: str, max_chars: int) -> List[str]:
        """按空行/换行优先切分，保证每段不超过 max_chars。"""
        text = text or ""
        if not text:
            return [""]
        if len(text) <= max_chars:
            return [text]

        chunks: List[str] = []
        # 先按空行分大段，再按行拼
        blocks = re.split(r"\n{2,}", text)
        buf = ""
        for block in blocks:
            candidate = block if not buf else buf + "\n\n" + block
            if len(candidate) <= max_chars:
                buf = candidate
                continue
            if buf:
                chunks.append(buf)
                buf = ""
            if len(block) <= max_chars:
                buf = block
                continue
            # 大段仍超长：按行
            lines = block.split("\n")
            line_buf = ""
            for line in lines:
                piece = line if not line_buf else line_buf + "\n" + line
                if len(piece) <= max_chars:
                    line_buf = piece
                    continue
                if line_buf:
                    chunks.append(line_buf)
                if len(line) <= max_chars:
                    line_buf = line
                else:
                    for i in range(0, len(line), max_chars):
                        chunks.append(line[i:i + max_chars])
                    line_buf = ""
            if line_buf:
                buf = line_buf
            else:
                buf = ""
        if buf:
            chunks.append(buf)
        return chunks or [text[:max_chars]]

    def _prepare_forward_text(self, text: str) -> str:
        """合并转发正文：按样式模式处理，并清洗易导致伪造转发失败的内容。"""
        mode = getattr(self, "qq_rich_text_mode", "sanitize") or "sanitize"
        if mode == "stylize":
            s = stylize_qq_text(text or "")
        elif mode == "off":
            s = text or ""
        else:
            s = sanitize_qq_text(text or "")

        # DeepWiki 杂质
        s = re.sub(r"<cite\b[^>]*/?>", "", s, flags=re.I)
        s = re.sub(r"</cite>", "", s, flags=re.I)
        s = re.sub(r"View this search on DeepWiki:.*", "", s, flags=re.I)
        s = re.sub(r"Wiki pages you might want to explore:[\s\S]*?(?=\n\n|\Z)", "", s, flags=re.I)
        # 控制字符
        s = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", s)
        # 数学字母平面（若仍有）→ 基本拉丁
        out = []
        for ch in s:
            o = ord(ch)
            if 0x1D400 <= o <= 0x1D7FF:
                if 0x1D400 <= o <= 0x1D419:
                    out.append(chr(ord("A") + (o - 0x1D400)))
                elif 0x1D41A <= o <= 0x1D433:
                    out.append(chr(ord("a") + (o - 0x1D41A)))
                elif 0x1D5D4 <= o <= 0x1D5ED:
                    out.append(chr(ord("A") + (o - 0x1D5D4)))
                elif 0x1D5EE <= o <= 0x1D607:
                    out.append(chr(ord("a") + (o - 0x1D5EE)))
                elif 0x1D608 <= o <= 0x1D621:
                    out.append(chr(ord("A") + (o - 0x1D608)))
                elif 0x1D622 <= o <= 0x1D63B:
                    out.append(chr(ord("a") + (o - 0x1D622)))
                elif 0x1D7CE <= o <= 0x1D7D7:
                    out.append(chr(ord("0") + (o - 0x1D7CE)))
                elif 0x1D7E2 <= o <= 0x1D7EB:
                    out.append(chr(ord("0") + (o - 0x1D7E2)))
                continue
            if 0xD800 <= o <= 0xDFFF:
                continue
            out.append(ch)
        s = "".join(out)
        s = re.sub(r"\n{3,}", "\n\n", s).strip()
        return s

    def _make_forward_node_variants(self, text: str, self_id: str, bot_nick: str) -> List[dict]:
        """生成多种节点格式，按兼容性依次尝试。"""
        name = str(bot_nick or "DeepWiki")[:32]
        uin = str(self_id)
        # 1) go-cqhttp 经典：字符串 content
        a = {"type": "node", "data": {"name": name, "uin": uin, "content": text}}
        # 2) content 为消息段数组
        b = {
            "type": "node",
            "data": {
                "name": name,
                "uin": uin,
                "content": [{"type": "text", "data": {"text": text}}],
            },
        }
        # 3) user_id + nickname（部分 OneBot 实现）
        try:
            uid = int(self_id)
        except Exception:
            uid = self_id
        c = {
            "type": "node",
            "data": {
                "user_id": uid,
                "nickname": name,
                "content": [{"type": "text", "data": {"text": text}}],
            },
        }
        return [a, b, c]

    def _build_forward_nodes(
        self, content: str, self_id: str, bot_nick: str, fmt_idx: int = 0
    ) -> List[dict]:
        """长文 → 多 node → 一次发送 = 同一个合并转发卡片。"""
        cleaned = self._prepare_forward_text(content)
        max_chars = getattr(self, "forward_node_max_chars", 800) or 800
        parts = self._split_text_chunks(cleaned, max_chars)
        total = len(parts)
        nodes: List[dict] = []
        for idx, part in enumerate(parts, 1):
            body = f"({idx}/{total})\n{part}" if total > 1 else part
            variants = self._make_forward_node_variants(body, self_id, bot_nick)
            nodes.append(variants[fmt_idx % len(variants)])
        return nodes

    def _build_forward_nodes_with_limit(
        self, content: str, self_id: str, bot_nick: str, max_chars: int, fmt_idx: int = 0
    ) -> List[dict]:
        cleaned = self._prepare_forward_text(content)
        parts = self._split_text_chunks(cleaned, max_chars)
        total = len(parts)
        nodes: List[dict] = []
        for idx, part in enumerate(parts, 1):
            body = f"({idx}/{total})\n{part}" if total > 1 else part
            variants = self._make_forward_node_variants(body, self_id, bot_nick)
            nodes.append(variants[fmt_idx % len(variants)])
        return nodes

    def _forward_fail_notice(
        self,
        err: Exception,
        node_count: int,
        char_count: int,
        api_timeout: float,
    ) -> str:
        err_s = str(err) or type(err).__name__
        likely = (
            f"NapCat 合并转发接口失败（请求超时设定 {api_timeout:.0f}s）。"
            "若错误含「伪造合并转发」：多为节点格式/特殊字符编码问题，或 NapCat 版本 bug，"
            "不一定是字数过多。"
        )
        if "超时" in err_s or "timeout" in err_s.lower():
            likely = (
                f"等待 NapCat 响应超时（{api_timeout:.0f}s）。"
                "也可能是群负载高；与字数无必然关系。"
            )
        if "伪造" in err_s or "res_id" in err_s:
            likely = (
                "NapCat 上传伪造合并转发失败（协议侧拒绝）。"
                "常见：节点字段不兼容、内容含特殊 Unicode、或 NapCat 版本问题。"
                "插件已改用 name+uin+纯文本 content 格式并剥离数学粗体字符。"
            )
        if "长度" in err_s or "too long" in err_s.lower() or "limit" in err_s.lower():
            likely = "消息长度超过平台限制"
        return (
            "【DeepWiki】合并转发失败\n"
            f"错误：{err_s[:240]}\n"
            f"可能原因：{likely}\n"
            f"本次：{node_count} 个节点，约 {char_count} 字，"
            f"单节点上限 {self.forward_node_max_chars} 字。\n"
            "长文会拆成多条假消息放进同一个合并转发卡片。"
            "可调大「合并转发 API 超时秒数」、调小「单节点字数」，"
            "或开启「转发失败后降级为普通消息」。"
        )

    async def _send_as_forward_content(self, event: KiraMessageEvent, content: str):
        """
        合并转发：长文 → 多 node → 一次 API（一个卡片）。
        失败原因也尽量用合并转发卡片发出。
        """
        adapter_name = event.adapter.name
        adapter_inst = self.ctx.adapter_mgr.get_adapter(adapter_name)
        if not adapter_inst or not hasattr(adapter_inst, "get_client"):
            await self._send_text_reply(event, content)
            return

        client = adapter_inst.get_client()
        # 非 OneBot/NapCat 客户端没有 send_action（如微信/TG/DC），直接走普通消息
        if not client or not hasattr(client, "send_action"):
            await self._send_text_reply(event, content)
            return

        is_group = event.message.group is not None
        session_id = str(event.session.session_id)
        # 关键：KiraMessageEvent 没有 .self_id，必须取 event.message.self_id（机器人 QQ）。
        # 之前 getattr(event, "self_id", "0") 恒为 "0"，转发节点 uin 无效，
        # 是 NapCat「伪造合并转发失败」的主要根因。
        self_id = str(
            getattr(event.message, "self_id", "")
            or getattr(event, "self_id", "")
            or "0"
        )
        bot_nick = getattr(adapter_inst.info, "name", None) or "DeepWiki"
        api_timeout = float(getattr(self, "forward_api_timeout", 60) or 60)

        prepared = self._prepare_forward_text(content)
        char_count = len(prepared)
        node_max = int(getattr(self, "forward_node_max_chars", 800) or 800)

        def _num(s: Any) -> Any:
            """group_id/user_id 尽量转 int；非数字（非 QQ 平台）则原样传入。"""
            try:
                return int(s)
            except (TypeError, ValueError):
                return s

        async def _do_forward(node_list: List[dict], timeout: float) -> None:
            import inspect
            kwargs = {}
            try:
                if "timeout" in inspect.signature(client.send_action).parameters:
                    kwargs["timeout"] = timeout
            except Exception:
                kwargs = {"timeout": timeout}

            async def _call(action: str, params: dict):
                if kwargs:
                    return await client.send_action(action, params, **kwargs)
                return await client.send_action(action, params)

            target_id = _num(session_id)
            if is_group:
                actions = [
                    ("send_group_forward_msg", {"group_id": target_id, "messages": node_list}),
                    ("send_forward_msg", {"group_id": target_id, "messages": node_list}),
                ]
            else:
                actions = [
                    ("send_private_forward_msg", {"user_id": target_id, "messages": node_list}),
                ]
                if self.enable_private_forward_fallback:
                    actions.append(
                        ("send_forward_msg", {"user_id": target_id, "messages": node_list})
                    )

            last_local_err: Optional[Exception] = None
            for action, params in actions:
                try:
                    resp = await _call(action, params)
                    if isinstance(resp, dict) and resp.get("status") == "failed":
                        msg = (
                            resp.get("message")
                            or resp.get("wording")
                            or resp.get("retcode")
                            or resp
                        )
                        last_local_err = RuntimeError(f"send_forward failed: {msg}")
                        logger.warning(f"{action} status=failed: {msg}")
                        continue
                    return
                except Exception as e:
                    last_local_err = e
                    logger.warning(f"{action} raised: {e}")
                    # 超时说明 NapCat 已受理但未及时回包，换 action 名无意义，直接上抛
                    if isinstance(e, (asyncio.TimeoutError, TimeoutError)) or "超时" in str(e):
                        raise
                    continue
            if last_local_err:
                raise last_local_err
            raise RuntimeError("send_forward failed: unknown")

        last_err: Optional[Exception] = None
        last_nodes: List[dict] = []

        # 依次尝试：3 种节点格式 → 更小切分 → 单节点截断（惰性构建，避免无谓拆分）
        def _attempt_plan() -> List[Tuple[str, Any]]:
            plan: List[Tuple[str, Any]] = []
            for fmt_idx in range(3):
                plan.append(
                    (
                        f"fmt{fmt_idx}-split",
                        lambda i=fmt_idx: self._build_forward_nodes(content, self_id, bot_nick, i),
                    )
                )
            half = max(400, node_max // 2)
            plan.append(
                (
                    "fmt0-half",
                    lambda: self._build_forward_nodes_with_limit(content, self_id, bot_nick, half, 0),
                )
            )

            def _one() -> List[dict]:
                one = prepared if len(prepared) <= 900 else prepared[:900] + "\n...(truncated)"
                return [self._make_forward_node_variants(one, self_id, bot_nick)[0]]

            plan.append(("fmt0-one", _one))
            return plan

        for idx, (label, build) in enumerate(_attempt_plan()):
            node_list = build()
            last_nodes = node_list
            # 首个尝试用完整超时；后续尝试递减，避免最坏情况十几分钟无响应
            attempt_timeout = api_timeout if idx == 0 else min(25.0, api_timeout)
            try:
                await _do_forward(node_list, attempt_timeout)
                logger.info(
                    f"DeepWiki forward ok: {label}, nodes={len(node_list)}, chars={char_count}"
                )
                return
            except Exception as e:
                logger.warning(f"Forward attempt {label} failed: {e}")
                last_err = e

        # 失败原因：优先合并转发卡片（短文本，用最简节点格式）
        notice = self._forward_fail_notice(
            last_err or RuntimeError("unknown"),
            len(last_nodes),
            char_count,
            api_timeout,
        )
        for fmt_idx in range(3):
            notice_nodes = [self._make_forward_node_variants(notice, self_id, bot_nick)[fmt_idx]]
            try:
                await _do_forward(notice_nodes, min(20.0, api_timeout))
                logger.info(f"Forward-fail notice sent as merge-forward card (fmt{fmt_idx})")
                break
            except Exception as e4:
                logger.error(f"Fail-notice card fmt{fmt_idx} failed: {e4}")
        else:
            try:
                await self._send_text_reply(event, notice)
            except Exception as e5:
                logger.error(f"Failed to send forward-fail notice as text: {e5}")

        if self.enable_forward_plain_fallback:
            try:
                # 普通消息仍走用户选择的样式模式
                await self._send_text_reply(event, content)
                logger.info("Forward failed; plain-text full fallback sent (switch on)")
            except Exception as e6:
                logger.error(f"Plain text full fallback failed: {e6}")
        else:
            logger.info(
                "Forward failed; plain-text full fallback skipped "
                "(enable_forward_plain_fallback=false)"
            )

    async def _execute_direct_query(
        self,
        event: KiraMessageEvent,
        repo: str,
        question: str,
        ctx_key: Optional[str] = None,
    ):
        p = self._primary_cmd()
        if not self._client:
            await self._send_maybe_forward(event, "DeepWiki 客户端未初始化")
            return

        key = self._cache_key(repo, question)
        cached = self._get_cached_answer(key)
        if cached:
            await self._send_deepwiki_result(event, repo, question, cached)
            return

        answer = await self._client.ask_question(repo, question)
        if self._is_repo_query_failure(answer):
            # 仓库未索引 / 查询失败：只清当前绑定仓库，保留候选列表便于 /dw 1 重选
            ck = ctx_key or self._get_ctx_key(event)
            self._last_repo.pop(ck, None)
            has_cands = bool(self._last_candidates.get(ck))
            tip = (
                f"查询失败或仓库未被 DeepWiki 索引：\n"
                f"{answer}\n\n"
                f"✅ 已取消当前仓库绑定（避免后续关键词被当成追问）。\n"
                f"请重新选择：\n"
                f"• {p} <关键词>     重新搜索\n"
                f"• {p} owner/repo   直接指定其他仓库"
            )
            if has_cands:
                tip += f"\n• {p} 1 / {p} 2     从刚才的候选列表重选"
            await self._send_maybe_forward(event, tip)
            return

        self._set_cached_answer(key, answer)
        await self._send_deepwiki_result(event, repo, question, answer)

    async def _send_deepwiki_result(self, event: KiraMessageEvent, repo: str, question: str, answer: str):
        """根据配置决定直接发文本还是合并转发"""
        if not self.enable_auto_forward:
            await self._send_text_reply(event, answer)
            return

        # force_forward_all 开启 → 答案必须走转发
        if self.force_forward_all:
            formatted = self._build_answer_with_metadata(repo, question, answer)
            await self._send_as_forward_content(event, formatted)
            return

        # 否则只有启用长度判断且超过阈值时才转发
        if self.use_length_threshold and len(answer) > self.forward_threshold:
            formatted = self._build_answer_with_metadata(repo, question, answer)
            await self._send_as_forward_content(event, formatted)
        else:
            await self._send_text_reply(event, answer)

    def _build_answer_with_metadata(self, repo: str, question: str, answer: str) -> str:
        """把仓库/问题元数据和答案组装成最终卡片内容，并在末尾（可选）附加操作指南。"""
        content = answer or ""
        if self.prepend_metadata_in_card:
            prefix = self._render_template(
                self.forward_metadata_template, repo=repo, question=question
            )
            if prefix:
                content = prefix + content

        if self.append_operation_guide:
            # 空 {} / 空字符串 → 自动生成；有自定义内容才用模板
            if is_empty_template(self.operation_guide_text):
                guide = self._get_operation_guide()
            else:
                guide = self._render_template(self.operation_guide_text, cmd=self._primary_cmd())
            if guide and guide.strip() and guide.strip() not in ("{}", "[]"):
                content = content.rstrip() + "\n\n" + guide.strip()

        return content

    async def _send_text_reply(self, event: KiraMessageEvent, text: str):
        """普通文本回复；过长时自动分段发送，避免单条消息失败。"""
        sid = self._get_sid(event)
        cleaned = self._sanitize(text)
        chunks = self._split_text_chunks(cleaned, self._PLAIN_MSG_MAX_CHARS)
        for i, chunk in enumerate(chunks):
            body = chunk if len(chunks) == 1 else f"[{i + 1}/{len(chunks)}]\n{chunk}"
            await self.ctx.send_message_chain(
                session=sid,
                chain=MessageChain([Text(body)]),
            )
            if i < len(chunks) - 1:
                await asyncio.sleep(0.35)

    async def _send_as_forward(self, event: KiraMessageEvent, repo: str, question: str, answer: str):
        """自实现合并转发逻辑（仅用于旧路径兜底，实际已由 _send_as_forward_content 接管）"""
        await self._send_as_forward_content(event, self._build_answer_with_metadata(repo, question, answer))

    def _get_sid(self, event) -> str:
        if hasattr(event, "sid"):
            return event.sid
        if hasattr(event, "session") and hasattr(event.session, "sid"):
            return event.session.sid
        # 兜底：自行从 adapter / group / sender 组装，
        # 避免某些 core 版本事件上没有 session 时全部落到 "default" 导致预设无法匹配
        try:
            adapter_name = getattr(getattr(event, "adapter", None), "name", "unknown")
            group = getattr(getattr(event, "message", None), "group", None)
            if group is not None:
                return f"{adapter_name}:gm:{getattr(group, 'group_id', 'unknown')}"
            sender = getattr(getattr(event, "message", None), "sender", None)
            if sender is not None:
                return f"{adapter_name}:dm:{getattr(sender, 'user_id', 'unknown')}"
        except Exception:
            pass
        return "default"

    def _get_ctx_key_batch(self, event) -> str:
        """KiraMessageBatchEvent（LLM 工具/llm_request 钩子）的上下文 key。
        - 默认按 event.sid 隔离
        - isolate_context_by_user=True 时按消息发送者 user_id 隔离
        """
        sid = getattr(event, "sid", None) or getattr(getattr(event, "session", None), "sid", "default")
        if getattr(self, "isolate_context_by_user", False):
            try:
                msgs = getattr(event, "messages", None) or []
                if msgs:
                    uid = getattr(getattr(msgs[-1], "sender", None), "user_id", None)
                    if uid is not None:
                        return f"user:{uid}"
            except Exception:
                pass
            last_part = sid.split(":")[-1] if ":" in sid else sid
            return f"user:{last_part}"
        return sid

    def _llm_bound_repo(self, event) -> Tuple[Optional[str], Optional[str]]:
        """LLM 路径当前生效的绑定仓库（已绑定优先，否则按分级预设）。返回 (ctx_key, repo)。"""
        ctx_key = self._get_ctx_key_batch(event)
        session_key = getattr(event, "sid", None) or getattr(
            getattr(event, "session", None), "sid", None
        )
        repo = self._last_repo.get(ctx_key) or self._resolve_preset_repo(ctx_key, session_key)
        return ctx_key, repo

    # ============================================================
    # LLM 请求注入：告知默认仓库，省去搜索环节
    # ============================================================

    @on.llm_request()
    async def inject_default_repo_prompt(self, event, req, *_):
        """开启「LLM 默认绑定预设仓库」后，在系统提示里告知当前默认仓库。"""
        if not self.enabled or not self.enable_llm_tool or not self.llm_bind_preset_repo:
            return
        try:
            _, repo = self._llm_bound_repo(event)
            if not repo:
                return
            note = (
                f"\n当前会话默认 DeepWiki 仓库：{repo}。"
                f"用户就本项目/该仓库提问时，直接调用 ask_deepwiki(repo=\"{repo}\", ...)，"
                "无需先 search_deepwiki；只有用户明确询问其他项目时才重新搜索。"
            )
            prompts = getattr(req, "system_prompt", None) or []
            for p in prompts:
                if getattr(p, "name", None) == "tools":
                    p.content = (getattr(p, "content", "") or "") + note
                    return
        except Exception as e:
            logger.warning(f"DeepWiki inject default repo prompt failed: {e}")

    def _get_ctx_key(self, event) -> str:
        """返回当前上下文的 key。
        - 默认按 sid（会话/群聊）隔离
        - 如果 isolate_context_by_user=True，则按真实用户 ID 隔离
        """
        if getattr(self, "isolate_context_by_user", False):
            # 优先从消息 sender 拿 user_id
            try:
                if hasattr(event, "message") and hasattr(event.message, "sender"):
                    uid = getattr(event.message.sender, "user_id", None)
                    if uid is not None:
                        return f"user:{uid}"
            except Exception:
                pass
            # 兜底：用 sid 最后一段
            sid = self._get_sid(event)
            last_part = sid.split(":")[-1] if ":" in sid else sid
            return f"user:{last_part}"
        else:
            return self._get_sid(event)

    def _find_preset_repo(self, ctx_key: str) -> Optional[str]:
        """查找该 ctx_key 配置了的预设仓库。

        ctx_key 形如 "adapter:gm:群号" / "adapter:dm:QQ" / "user:QQ"。
        匹配规则（从上到下第一个命中）：
        1. 完全相等
        2. 忽略适配器名：pattern 与 ctx_key 的「类型:ID」尾段一致
           （适配器名是用户自定义的，预设里写错名字也能匹配上）
        3. pattern 为 "gm:群号" / 裸 ID 等形式，命中 ctx_key 尾段
        4. 兼容旧版前缀匹配
        """
        if not ctx_key:
            return None
        parts = ctx_key.split(":")
        tail2 = ":".join(parts[-2:]) if len(parts) >= 2 else ctx_key
        tail1 = parts[-1]
        for pattern, repo in self.default_repo_presets:
            p = (pattern or "").strip()
            if not p:
                continue
            if p == ctx_key:
                return repo
            p_parts = p.split(":")
            if len(p_parts) >= 3 and len(parts) >= 3 and p_parts[-2:] == parts[-2:]:
                return repo
            if p == tail2 or p == tail1:
                return repo
            if ctx_key.startswith(p) or p.startswith(ctx_key):
                return repo
        return None

    def _resolve_preset_repo(self, ctx_key: str, session_key: Optional[str] = None) -> Optional[str]:
        """分级解析预设仓库：
        1. 先按 ctx_key 精确匹配（用户隔离模式下即 user:QQ 专属预设）
        2. 未命中且处于用户隔离模式（ctx_key != session_key）时，
           回落到会话级预设 —— 会话级预设接管该会话内所有用户
        """
        repo = self._find_preset_repo(ctx_key)
        if repo:
            return repo
        if session_key and session_key != ctx_key:
            return self._find_preset_repo(session_key)
        return None

    def _apply_default_preset_if_needed(self, ctx_key: str, session_key: Optional[str] = None) -> Optional[str]:
        """为 ctx_key 应用预设仓库（已绑定时不覆盖）。返回最终生效的预设或 None。"""
        preset = self._resolve_preset_repo(ctx_key, session_key)
        if preset and ctx_key not in self._last_repo:
            self._last_repo[ctx_key] = preset
            logger.info(f"DeepWiki: ctx {ctx_key} 应用预设仓库 {preset}（会话 {session_key}）")
        return preset

    def _clear_ctx(self, ctx_key: str):
        """清除指定上下文的仓库记忆和候选列表。"""
        self._last_repo.pop(ctx_key, None)
        self._last_candidates.pop(ctx_key, None)

    def _is_clear_intent(self, text: str) -> bool:
        """整词/整句匹配清除意图，避免误伤正常问题。"""
        if not text:
            return False
        t = text.lower().strip()
        for word in self.clear_command_words:
            if not word:
                continue
            w = word.lower().strip()
            if not w:
                continue
            if t == w or t.startswith(w + " ") or t.endswith(" " + w):
                return True
        extra = {"forget", "忘掉", "不要记住", "别记了", "清除上下文", "重置上下文"}
        return t in extra

    def _is_status_intent(self, text: str) -> bool:
        """整词匹配状态查询。"""
        if not text:
            return False
        t = text.strip().lower()
        for word in self.status_command_words:
            if not word:
                continue
            w = word.lower().strip()
            if w and t == w:
                return True
        return False

    def _get_operation_guide(self) -> str:
        """根据当前配置动态生成完整操作指南（命令词带前缀）。

        仅修正说明括号文案；其余条目完整保留。
        """
        p = self._primary_cmd()
        triggers = [w for w in (self.command_words or []) if w][:3]
        clears = [w for w in (self.clear_command_words or []) if w][:3]
        statuses = [w for w in (self.status_command_words or []) if w][:3]

        lines = ["——操作指南——", f"【{p} 操作指南】"]

        if triggers:
            t = " / ".join(triggers)
            lines.append(f"• {t} <关键词或问题>   （触发查询）")

        if statuses:
            s = " / ".join(f"{p} {w}" for w in statuses)
            lines.append(f"• {s}   （查看当前上下文仓库）")

        if clears:
            # 括号文案按用户要求
            c = " / ".join(f"{p} {w}" for w in clears)
            lines.append(f"• {c}   （清除上下文，重新查询仓库来提问）")

        lines.append(f"• {p} 1 / {p} 2   （从候选列表中查询对应序号的仓库）")
        lines.append(f"• {p} <问题>   （在已有上下文仓库上继续追问）")
        lines.append(
            f"• 切换项目：{p} clear 后，再 {p} <新关键词>；"
            f"或在无上下文时直接 {p} <新关键词> 搜索"
        )
        lines.append(f"• 也可直接 {p} owner/repo 指定仓库")

        return "\n".join(lines)

    # ============================================================
    # LLM 工具（自然语言路径）
    # ============================================================

    @tool(
        name="search_deepwiki",
        description="搜索 GitHub 仓库，返回 owner/repo 格式的候选列表。当用户询问某个项目但未提供完整仓库路径时，必须使用此工具获取准确的项目标识。支持多路融合搜索，返回多个候选供选择。",
        params={
            "type": "object",
            "properties": {
                "keyword": {
                    "type": "string",
                    "description": "项目关键词，例如 'KiraAI' 或 'react'",
                }
            },
            "required": ["keyword"],
        },
    )
    async def search_deepwiki(self, event: KiraMessageBatchEvent, keyword: str) -> str:
        if not self.enable_llm_tool:
            return "LLM 工具调用已被禁用，请使用 /dw 命令直接查询。"

        cached = self._get_cached_search(keyword)
        if cached is None:
            if self.use_multi_path:
                cached = await _multi_path_search_repositories(keyword, self.github_token, self.max_search_results)
            else:
                # 回退到单路
                items = await _github_search(keyword, self.github_token, per_page=self.max_search_results)
                cached = []
                for item in items:
                    cached.append({
                        "full_name": item.get("full_name"),
                        "stars": item.get("stargazers_count", 0),
                        "description": item.get("description", ""),
                    })
            self._set_cached_search(keyword, cached)

        raw = _format_repo_candidates(cached, cmd_prefix=self._primary_cmd())
        # 「LLM 默认绑定预设仓库」开启：提示 LLM 可直接使用默认仓库
        if self.llm_bind_preset_repo:
            _, bound = self._llm_bound_repo(event)
            if bound:
                raw += (
                    f"\n\n提示：当前会话默认仓库 {bound}。"
                    f"与本项目相关的问题可直接 ask_deepwiki(repo=\"{bound}\", ...)，无需从候选中选择。"
                )
        return self._sanitize(raw)

    @tool(
        name="ask_deepwiki",
        description="向 DeepWiki 提问关于 GitHub 仓库的问题。**repo 参数必须是 owner/repo 格式（例如 xxynet/KiraAI）**。如果你不知道准确的仓库路径，请先调用 search_deepwiki 工具搜索。",
        params={
            "type": "object",
            "properties": {
                "repo": {
                    "type": "string",
                    "description": "GitHub 仓库标识，格式 owner/repo，例如 xxynet/KiraAI",
                },
                "question": {
                    "type": "string",
                    "description": "用户的问题，例如“如何安装插件？”",
                },
            },
            "required": ["repo", "question"],
        },
    )
    async def ask_deepwiki(
        self, event: KiraMessageBatchEvent, repo: str, question: str
    ) -> str:
        if not self.enabled:
            return "DeepWiki 查询未启用"
        if not self._client:
            return "DeepWiki 客户端未初始化"
        if not self.enable_llm_tool:
            return "LLM 工具调用已被禁用，请使用 /dw 命令直接查询。"

        # 「LLM 默认绑定预设仓库」开启：repo 参数非法时用绑定/预设仓库兜底，成功后记录绑定
        bind_ctx = None
        if self.llm_bind_preset_repo:
            bind_ctx, bound = self._llm_bound_repo(event)
            if bound and not re.match(r"^[\w.-]+/[\w.-]+$", (repo or "").strip()):
                logger.info(f"DeepWiki LLM: repo 参数无效({repo!r})，改用绑定仓库 {bound}")
                repo = bound

        key = self._cache_key(repo, question)
        cached = self._get_cached_answer(key)
        if cached:
            return cached

        answer = await self._client.ask_question(repo, question)
        if self._is_repo_query_failure(answer):
            return (
                f"未找到相关信息或仓库未被 DeepWiki 索引：{answer}\n"
                "请换一个已索引的 owner/repo，或先 search_deepwiki 重新搜索。"
            )

        self._set_cached_answer(key, answer)
        if bind_ctx:
            self._last_repo[bind_ctx] = repo
        return answer

    @tool(
        name="get_deepwiki_structure",
        description="获取指定 GitHub 仓库的 DeepWiki 文档结构（目录/主题列表）。用于了解仓库有哪些文档页面，方便后续针对性查询。repo 必须是 owner/repo 格式。",
        params={
            "type": "object",
            "properties": {
                "repo": {
                    "type": "string",
                    "description": "GitHub 仓库标识，格式 owner/repo，例如 xxynet/KiraAI",
                }
            },
            "required": ["repo"],
        },
    )
    async def get_deepwiki_structure(self, event: KiraMessageBatchEvent, repo: str) -> str:
        if not self.enabled:
            return "DeepWiki 查询未启用"
        if not self._client:
            return "DeepWiki 客户端未初始化"
        if not self.enable_llm_tool:
            return "LLM 工具调用已被禁用，请使用 /dw 命令直接查询。"
        return await self._client.read_wiki_structure(repo)

    @tool(
        name="read_deepwiki_content",
        description="读取指定 GitHub 仓库的 DeepWiki 文档内容。可选指定主题（建议先用 get_deepwiki_structure 获取主题列表）。repo 必须是 owner/repo 格式。topic 留空则返回整体概览或首页内容。",
        params={
            "type": "object",
            "properties": {
                "repo": {
                    "type": "string",
                    "description": "GitHub 仓库标识，格式 owner/repo，例如 xxynet/KiraAI",
                },
                "topic": {
                    "type": "string",
                    "description": "可选主题名称，如 'Installation'、'Architecture' 等，留空返回整体内容",
                },
            },
            "required": ["repo"],
        },
    )
    async def read_deepwiki_content(
        self, event: KiraMessageBatchEvent, repo: str, topic: str = ""
    ) -> str:
        if not self.enabled:
            return "DeepWiki 查询未启用"
        if not self._client:
            return "DeepWiki 客户端未初始化"
        if not self.enable_llm_tool:
            return "LLM 工具调用已被禁用，请使用 /dw 命令直接查询。"

        key = self._cache_key(repo, f"content:{topic}")
        cached = self._get_cached_answer(key)
        if cached:
            return cached

        answer = await self._client.read_wiki_contents(repo, topic)
        if not answer or "error" in answer.lower() or "failed" in answer.lower():
            return f"读取内容失败：{answer}"

        self._set_cached_answer(key, answer)
        return answer
