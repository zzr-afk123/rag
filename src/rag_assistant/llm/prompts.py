"""提示词集中管理。

三个约定，务必保持：
1. **任务标记**：system 消息以 ``[[TASK:xxx]]`` 开头。``MockLLM`` 靠它分派行为，
   否则 mock 无法在无网络环境下模拟"改写/生成/摘要"三种不同输出。
2. **只输出 JSON** 的任务（改写）要在 prompt 里写死 schema，解析端还要有兜底
   （见 ``generation/rewriter.py``）——LLM 不遵守格式是常态，不是异常。
3. **拒答话术与免责声明不写进 prompt 让模型"自由发挥"**：拒答话术由代码替换、
   免责声明由代码追加。prompt 里只告诉模型"资料不足时输出这句话"，
   而且最终输出还要过校验（见 ``generation/answerer.py`` 的两道闸门）。
"""

from __future__ import annotations

from typing import Sequence

from ..llm.base import LLMMessage, system_message, user_message

TASK_REWRITE = "rewrite"
TASK_ANSWER = "answer"
TASK_SUMMARY = "summary"
TASK_JUDGE = "judge"

_MARKER = "[[TASK:{task}]]"


def _system(task: str, body: str) -> str:
    return f"{_MARKER.format(task=task)}\n{body.strip()}"


# --------------------------------------------------------------------------
# 问题改写（指代消解 + Multi-Query）
# --------------------------------------------------------------------------
REWRITE_SYSTEM = _system(
    TASK_REWRITE,
    """
你是"制度与办事流程"问答系统的检索查询改写器。你的唯一任务是把用户问题改写成适合检索的查询。

要求：
1. standalone：结合历史对话，把代词（它、这个、那个、该、上述、此、其中）替换成具体对象，
   输出一个**完整、独立、可单独检索**的问题。若当前问题本身已完整，则原样保留其关键信息。
2. variants：生成 2-3 条与 standalone 语义等价但表述不同的检索查询，用于多路召回
   （例如补充同义说法、把口语词换成制度用语）。不得引入用户没有提及的新条件。
3. 不要回答问题，不要解释，不要输出任何思考过程。
4. 只输出一个 JSON 对象，格式严格如下：
   {"standalone": "……", "variants": ["……", "……"]}
""",
)


def rewrite_prompt(question: str, history_text: str, variant_count: int) -> list[LLMMessage]:
    """构造改写请求。

    Args:
        question: 用户当前问题（可能带指代）。
        history_text: 最近若干轮对话的纯文本（空串表示首轮）。
        variant_count: 期望生成的等价查询条数。
    """
    parts: list[str] = []
    if history_text.strip():
        parts.append(f"历史对话：\n{history_text.strip()}")
    else:
        parts.append("历史对话：（无，这是本会话第一轮）")
    parts.append(f"当前问题：{question}")
    parts.append(f"请生成 {max(1, variant_count)} 条 variants，只输出 JSON。")
    return [system_message(REWRITE_SYSTEM), user_message("\n\n".join(parts))]


# --------------------------------------------------------------------------
# 回答生成
# --------------------------------------------------------------------------
def answer_system(refuse_text: str) -> str:
    return _system(
        TASK_ANSWER,
        f"""
你是学院"制度与办事流程"智能问答助手。你只能依据下面提供的【资料】回答，不得使用资料之外的知识。

硬性规则：
1. 每一条关键结论后面必须标注来源编号，格式为 [1]、[2]（对应资料编号）。可以一条结论标多个编号。
   编号要紧跟在句末（如"……逾期按退学处理[1]。"），不要单独占一行。
2. 严禁编造制度条款、数字、期限、金额、部门名称。资料里没写的，就说资料未提及，不要推测。
3. 如果资料完全不涉及该问题，只输出这一句话，不要加任何其他内容：
   {refuse_text}
3.1 如果资料只覆盖了问题的一部分，就只回答资料明确写到的内容（照第 1 条标注编号），
   不要额外加"资料未提及""其余部分资料未提及"这类边界说明；
   完全没有任何相关内容时，用第 3 条的整句拒答。
   例：问"复学需要提交什么材料"，资料只写了"因病休学者须提交二级甲等以上医院的康复证明"，
   就答"因病休学者须提交二级甲等以上医院的康复证明[1]。"，不要拒答，也不要补其他材料。
4. 不要输出"参考资料""免责声明""以上内容仅供参考"之类的套话，系统会自动追加。
4.1 不要在答案末尾自己列"引用来源 / 参考资料"清单——页面会按编号自动生成引用卡片，
    你只要把编号写在句子末尾即可；表格也要改写成句子（如"会议费每人每天 500 元"），
    不要把竖线表格原样抄进来。
5. 用中文回答，简洁、分点，先给结论再给依据要点；不要复述整段原文。
6. **资料与问题无关时（例如问天气、闲聊、问其他单位的事），必须按第 3 条拒答**，
   不要为了用上资料而强行回答。
7. 每个分点句末必须标注其依据编号；一句话里若混用了两个章节的内容，
   必须分别标注（如"[1][3]"），不得只标其中一个。
8. **任何在所给资料里找不到编号的结论，直接不要写**——宁可不答，也不能无依据地补充。
""",
    )


def answer_prompt(
    question: str,
    context_blocks: Sequence[str],
    *,
    refuse_text: str,
    history_text: str = "",
) -> list[LLMMessage]:
    """构造回答请求。

    ``context_blocks`` 已由调用方编号（``[1] ...`` 开头），数量与 token 预算受
    ``generation.max_context_chunks / max_context_tokens`` 控制。
    """
    context = "\n\n".join(context_blocks) if context_blocks else "（无相关资料）"
    parts = [f"【资料】\n{context}", f"【问题】\n{question}"]
    if history_text.strip():
        parts.insert(1, f"【最近对话】\n{history_text.strip()}")
    parts.append(
        "请严格按规则作答：每条结论带 [编号]；资料不足则只输出指定的拒答话术。"
    )
    return [system_message(answer_system(refuse_text)), user_message("\n\n".join(parts))]


# --------------------------------------------------------------------------
# 会话摘要压缩
# --------------------------------------------------------------------------
SUMMARY_SYSTEM = _system(
    TASK_SUMMARY,
    """
你是对话摘要器。把"较早的对话"压缩成不超过 150 字的中文要点，供后续轮次理解上下文。

要求：
1. 保留用户关注的具体对象（文档名、事项名、部门、期限、金额等实体）。
2. 保留已经给出的结论要点，删掉寒暄与重复。
3. 不要加引用编号，不要输出"摘要："之类的前缀，直接给要点。
""",
)


def summary_prompt(previous_summary: str, turns_text: str) -> list[LLMMessage]:
    """构造摘要请求。"""
    parts: list[str] = []
    if previous_summary.strip():
        parts.append(f"已有摘要：\n{previous_summary.strip()}")
    parts.append(f"待压缩的对话：\n{turns_text.strip()}")
    return [system_message(SUMMARY_SYSTEM), user_message("\n\n".join(parts))]


def task_of(messages: Sequence[LLMMessage]) -> str:
    """从消息里取出任务标记（mock 用；真实模型会忽略它）。"""
    for message in messages:
        if message.role == "system" and message.content.startswith("[[TASK:"):
            return message.content[len("[[TASK:") :].split("]]", 1)[0]
    return ""


# --------------------------------------------------------------------------
# 忠实度评审（阶段 F：LLM-as-judge）
# --------------------------------------------------------------------------
JUDGE_SYSTEM = _system(
    TASK_JUDGE,
    """
你是严格的事实核验员。给你一个问题、一个回答，以及该回答引用的原文片段。

请判断：**回答中的每一条事实性结论，是否都能在给出的原文片段里找到直接依据**。

判定规则：
1. 数字、期限、金额、部门名称必须与原文完全一致（"约""左右"不算一致）。
2. 回答里出现原文没有的条件、限制或推论 → 不忠实。
3. 只有"综合归纳"但每个要点都能在原文找到 → 仍算忠实。
4. 不要评价回答是否完整、是否有用，只看"有没有依据"。

只输出一个 JSON 对象：
{"faithful": true 或 false, "unsupported": ["无依据的结论", "..."], "reason": "一句话理由"}
""",
)


def judge_prompt(question: str, answer: str, cited_texts: Sequence[str]) -> list[LLMMessage]:
    """构造忠实度评审请求。"""
    context = "\n\n".join(f"[{index}] {text}" for index, text in enumerate(cited_texts, start=1))
    payload = f"【问题】\n{question}\n\n【回答】\n{answer}\n\n【引用的原文片段】\n{context or '（无）'}"
    return [system_message(JUDGE_SYSTEM), user_message(payload)]
