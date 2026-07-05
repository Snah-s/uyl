import re


class AssemblyAgent:
    """
    Agente de planning para Qwen3-8B.

    Idea central
    ------------
    Cada `scenario_context` YA es un prompt one-shot: contiene las reglas del
    dominio + UN ejemplo resuelto (en lenguaje natural) + el problema real cuyo
    ultimo bloque [PLAN] esta vacio.

    Historial de corridas (mystery, 30 casos de Examples.json)
    ----------------------------------------------------------
    * v1 (salida directa, "imita el ejemplo, no expliques"): 0.62/10, 124.7s.
      Fallo dominante: ~16/30 planes arrancaban con `(overcome ...)`/`(succumb ...)`,
      IMPOSIBLES desde el estado inicial (exigen `Pain`; el inicial tiene `Harmony`
      sin `Pain`). El optimo SIEMPRE empieza por `attack` o `feast`.
    * v2 (CoT con simulacion de estado + marcador `FINAL PLAN:`): 0.57/10, 2205s.
      REGRESION. El razonamiento en el canal de respuesta se comporta como
      thinking: explota el tiempo (73s/caso -> ~60 min sobre las 50 tareas) y
      degenera a temp 0 (planes vacios por agotar tokens; bucles y acciones
      duplicadas `[feast a b, feast a b, ...]`). PERO arreglo el idx-0: casi
      ningun plan empieza ya con overcome/succumb.

    Estrategia v3 (esta)
    --------------------
    Nos quedamos con lo unico que ayudo (las reglas de precondicion como
    RESTRICCIONES DURAS) y descartamos el razonamiento verboso: salida DIRECTA y
    corta (como v1, ~4s/caso), `max_new_tokens` bajo para cortar bucles, y una red
    de seguridad en el parser contra la duplicacion consecutiva (que en estos
    dominios nunca es un plan valido).

    Dominios
    --------
    - mystery/objects : attack/succumb (1 arg), feast/overcome (2 args)
    - blocks          : engage_payload/release_payload (1), mount_node/unmount_node (2)
    """

    # Marcador opcional: si el modelo lo emitiese, el parser toma solo lo de despues.
    FINAL_MARKER = "final plan:"

    SYSTEM_PROMPT = (
        "You are an expert STRIPS planner. The problem gives the domain rules, ONE "
        "solved example, and a final problem whose last [PLAN] is empty. Output "
        "ONLY the plan for the FINAL problem: one action per line, using the SAME "
        "wording as the example, then a final line [PLAN END]. No explanations, no "
        "numbering, no thinking, and never repeat an action twice in a row. Make "
        "the plan as SHORT as possible and stop as soon as the goal holds.\n"
        "\n"
        "Respect these preconditions of the mystery domain:\n"
        "- Attack X needs Province X + Planet X + Harmony (it produces Pain X).\n"
        "- Succumb X needs Pain X. Overcome X from Y needs Pain X + Province Y.\n"
        "- Feast X from Y needs 'X craves Y' + Province X + Harmony.\n"
        "- The initial state has Harmony but NO Pain, so the FIRST action is ALWAYS "
        "Attack or Feast, NEVER Overcome or Succumb.\n"
        "- To make 'X craves Y', finish with 'overcome X from Y' (which needs "
        "Pain X, so 'attack X' shortly before).\n"
        "- If X already craves something you need to change, do 'feast X from ...' "
        "first to clear it."
    )

    def __init__(self):
        self.system_prompt = self.SYSTEM_PROMPT

    # ------------------------------------------------------------------ #
    # API principal
    # ------------------------------------------------------------------ #
    def solve(self, scenario_context: str, llm_engine_func) -> list:
        respuesta = llm_engine_func(
            prompt=scenario_context,
            system=self.system_prompt,
            # Plan corto (<=6 acciones ~ 6 lineas). Un tope bajo mantiene ~4s/caso
            # y CORTA los bucles degenerados que aparecian con 1024 tokens.
            max_new_tokens=256,
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
            # thinking desactivado a proposito: dispara el tiempo (>2 min).
            enable_thinking=False,
        )

        dominio = self._detectar_dominio(scenario_context)
        return self._parsear_plan(respuesta, dominio)

    # ------------------------------------------------------------------ #
    # Deteccion de dominio
    # ------------------------------------------------------------------ #
    @staticmethod
    def _detectar_dominio(ctx: str) -> str:
        if "mount_node" in ctx or "block" in ctx.lower():
            return "blocks"
        return "mystery"

    # ------------------------------------------------------------------ #
    # Aislar el bloque del plan final (tras el razonamiento)
    # ------------------------------------------------------------------ #
    def _bloque_final(self, texto: str) -> str:
        low = texto.lower()

        # 1) Preferimos lo que venga tras el ULTIMO marcador "FINAL PLAN:".
        idx = low.rfind(self.FINAL_MARKER)
        if idx != -1:
            return texto[idx + len(self.FINAL_MARKER):]

        # 2) Si uso modo thinking, saltamos el bloque <think>...</think>.
        cierre = low.rfind("</think>")
        if cierre != -1:
            return texto[cierre + len("</think>"):]

        # 3) Sin marcadores: usamos el texto completo (fallback).
        return texto

    # ------------------------------------------------------------------ #
    # Parseo: texto del modelo -> lista de acciones canonicas
    # ------------------------------------------------------------------ #
    def _parsear_plan(self, texto: str, dominio: str) -> list:
        bloque = self._bloque_final(texto)
        acciones = []
        for linea in bloque.splitlines():
            l = linea.strip()
            if not l:
                continue
            low = l.lower()

            # Cerramos el plan cuando el modelo lo marca o abre otro problema.
            if "[plan end]" in low or "[statement]" in low:
                break

            accion = self._linea_a_canonico(low, dominio)
            # Red de seguridad: en mystery/blocks una accion nunca es valida dos
            # veces seguidas (su precondicion desaparece tras ejecutarla). Si el
            # modelo degenera y la repite, colapsamos el duplicado consecutivo.
            if accion and (not acciones or acciones[-1] != accion):
                acciones.append(accion)

        return acciones

    def _linea_a_canonico(self, low: str, dominio: str):
        # 1) Si el modelo ya escribio formato canonico, lo reutilizamos.
        m = re.search(r"\(\s*([a-z_]+)\s+([a-z0-9_ ]+?)\s*\)", low)
        if m:
            verbo, args = m.group(1), m.group(2).split()
            if self._verbo_valido(verbo, len(args), dominio):
                return "(" + " ".join([verbo] + args) + ")"

        # 2) Conversion desde lenguaje natural segun el dominio.
        if dominio == "mystery":
            return self._mystery_natural(low)
        return self._blocks_natural(low)

    # ---- mystery / objects -------------------------------------------- #
    @staticmethod
    def _mystery_natural(low: str):
        m = re.search(r"\bfeast\s+object\s+(\w+)\s+from\s+object\s+(\w+)", low)
        if m:
            return f"(feast {m.group(1)} {m.group(2)})"
        m = re.search(r"\bovercome\s+object\s+(\w+)\s+from\s+object\s+(\w+)", low)
        if m:
            return f"(overcome {m.group(1)} {m.group(2)})"
        m = re.search(r"\battack\s+object\s+(\w+)", low)
        if m:
            return f"(attack {m.group(1)})"
        m = re.search(r"\bsuccumb\s+object\s+(\w+)", low)
        if m:
            return f"(succumb {m.group(1)})"
        return None

    # ---- blocks ------------------------------------------------------- #
    @staticmethod
    def _blocks_natural(low: str):
        # unmount_node ANTES que mount_node (uno es subcadena del otro).
        m = re.search(
            r"unmount_node\s+the\s+(\w+)\s+block\s+from\s+on\s+top\s+of\s+the\s+(\w+)\s+block",
            low,
        )
        if m:
            return f"(unmount_node {m.group(1)} {m.group(2)})"
        m = re.search(
            r"mount_node\s+the\s+(\w+)\s+block\s+on\s+top\s+of\s+the\s+(\w+)\s+block",
            low,
        )
        if m:
            return f"(mount_node {m.group(1)} {m.group(2)})"
        m = re.search(r"pick\s+up\s+the\s+(\w+)\s+block", low)
        if m:
            return f"(engage_payload {m.group(1)})"
        m = re.search(r"put\s+down\s+the\s+(\w+)\s+block", low)
        if m:
            return f"(release_payload {m.group(1)})"
        return None

    # ------------------------------------------------------------------ #
    @staticmethod
    def _verbo_valido(verbo: str, nargs: int, dominio: str) -> bool:
        unarios = {"attack", "succumb"} if dominio == "mystery" else {"engage_payload", "release_payload"}
        binarios = {"feast", "overcome"} if dominio == "mystery" else {"mount_node", "unmount_node"}
        return (verbo in unarios and nargs == 1) or (verbo in binarios and nargs == 2)
