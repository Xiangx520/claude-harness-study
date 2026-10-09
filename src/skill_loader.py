import yaml

from pathlib import Path





class SkillLoader:
    def __init__(self, skills_dir: Path):
        self.skills_dir = skills_dir
        self.skills: dict[str, dict[str, str]] = {}
        self.scan()

    """
    一份标准的skill.md格式开头两行叫frontmatter（头部元数据）例如：
    ---
    name: xxx
    description: xxxx
    ---
    以下方法就是用于提取该metadata，和正文body
    """
    @staticmethod
    def parse_frontmatter(text: str) -> tuple[dict, str]:
        # 把文本拆成行 并保留每一行末尾换行符 便于后续把lines拼成body时保持文本结构
        lines = text.splitlines(keepends=True)
        # 检查第一行是否是---
        if not lines or lines[0].rstrip("\r\n") != "---":
            return {}, text

        # 检查是否有收束的--- 如果存在则取其所在序号的下一个序号
        closing_index = next(
            (index for index, line in enumerate(lines[1:], start=1)
             if line.rstrip("\r\n") == "---"),
            None,
        )
        if closing_index is None:
            return {}, text

        # 取出---包裹的内容
        frontmatter = "".join(lines[1:closing_index])
        # 取出正文内容
        body = "".join(lines[closing_index + 1:]).strip()
        try:
            metadata = yaml.safe_load(frontmatter) or {}
        except yaml.YAMLError:              # yaml解析失败报错
            metadata = {}
        if not isinstance(metadata, dict):  # 解析结果不是json报错
            metadata = {}
        return metadata, body


    # 把md的名字，描述，正文加载到skills列表里
    def scan(self):
        self.skills.clear()
        if not self.skills_dir.exists():
            return

        # 把skill仓库位置转为绝对路径
        skills_root = self.skills_dir.resolve()
        # 从skill仓库中找md文档
        for manifest in sorted(self.skills_dir.glob("*/SKILL.md")):
            if (not manifest.is_file()
                    or not manifest.resolve().is_relative_to(skills_root)):
                continue
            content = manifest.read_text(encoding="utf-8")
            # 解析头部元数据和正文
            metadata, body = self.parse_frontmatter(content)
            raw_name = metadata.get("name")
            name = raw_name.strip() if isinstance(raw_name, str) else ""
            name = name or manifest.parent.name
            raw_description = metadata.get("description")
            description = (raw_description.strip()
                           if isinstance(raw_description, str) else "")
            description = description or body.split("\n", 1)[0]
            description = " ".join(str(description).lstrip("# ").split())
            self.skills[name] = {
                "name": name,
                "description": description,
                "content": content,
            }

    # 把skill的名字描述打包成字符串 后续传给llm
    def catalog(self) -> str:
        if not self.skills:
            return "(no skills found)"
        return "\n".join(
            f"- {skill['name']}: {skill['description']}"
            for skill in self.skills.values()
        )

    # 加载skill正文
    def load(self, name: str) -> str:
        skill = self.skills.get(name)
        if skill:
            return skill["content"]
        # 如果加载的skill不存在 返回可选的所有skills列表
        available = ", ".join(self.skills) or "none"
        return f"Error: Unknown skill '{name}'. Available: {available}"