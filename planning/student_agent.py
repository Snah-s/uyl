import re


class AssemblyAgent:
    """
    Agente de planning para Qwen3-8B.

    Idea central
    ------------
    Cada `scenario_context` YA es un prompt one-shot: contiene las reglas del
    dominio + UN ejemplo resuelto (en lenguaje natural) + el problema real cuyo
    ultimo bloque [PLAN] esta vacio.

    Leccion de la corrida baseline (0.62/10, mystery)
    -------------------------------------------------
    El parseo ya es 100% fiable; lo que falla es el RAZONAMIENTO del modelo:
      * En ~16/30 casos el plan arrancaba con `(overcome ...)` o `(succumb ...)`,
        que son IMPOSIBLES desde el estado inicial (exigen `Pain`, y el estado
        inicial tiene `Harmony` sin `Pain`). El optimo SIEMPRE empieza por
        `attack` o `feast`. El modelo no chequeaba precondiciones.
      * Planes demasiado largos (8 acciones vs 4 optimas) => perdia el +2 de
        longitud del evaluador.
    Causa raiz: el prompt anterior PROHIBIA razonar ("no expliques") y solo pedia
    "imitar el estilo del ejemplo". Para un dominio ofuscado (Mystery-Blocksworld)
    eso copia la forma sin simular el estado.

    Estrategia nueva
    ----------------
    Convertimos el system en un CoT con SIMULACION DE ESTADO y chequeo de
    precondiciones, y le pedimos que emita el plan FINAL tras un marcador unico
    (`FINAL PLAN:`). El parser ignora todo el razonamiento y extrae solo ese
    bloque final, lo convierte a canonico y valida por aridad.

    Dominios
    --------
    - mystery/objects : attack/succumb (1 arg), feast/overcome (2 args)
    - blocks          : engage_payload/release_payload (1), mount_node/unmount_node (2)
    """

    # Marcador que separa el razonamiento del plan definitivo.
    FINAL_MARKER = "final plan:"

    SYSTEM_PROMPT = (
        "You are an expert automated planner for a STRIPS-style domain. You are "
        "given the domain rules, ONE solved example, and a new problem whose final "
        "[PLAN] is empty. Solve ONLY the final problem.\n"
        "\n"
        "Reason step by step BEFORE writing the plan:\n"
        "1. Write down the initial facts and the goal facts of the FINAL problem.\n"
        "2. Simulate the world. You may pick an action ONLY if its preconditions "
        "are currently true. After each action, update the facts "
        "(Province / Planet / Harmony / Pain / Craves) accordingly.\n"
        "3. Choose the SHORTEST sequence that makes ALL goal facts true, and STOP "
        "as soon as the goal holds. Do not add extra actions.\n"
        "\n"
        "Action rules (mystery domain):\n"
        "- Attack X: needs Province X, Planet X, Harmony -> +Pain X; "
        "-Province X, -Planet X, -Harmony.\n"
        "- Succumb X: needs Pain X -> +Province X, +Planet X, +Harmony; -Pain X.\n"
        "- Overcome X from Y: needs Province Y, Pain X -> +Harmony, +Province X, "
        "+'X craves Y'; -Province Y, -Pain X.\n"
        "- Feast X from Y: needs 'X craves Y', Province X, Harmony -> +Pain X, "
        "+Province Y; -'X craves Y', -Province X, -Harmony.\n"
        "KEY: a state that has Harmony but no Pain can NEVER start with Overcome or "
        "Succumb; the first action is always Attack or Feast.\n"
        "To make 'X craves Y' you need Overcome X from Y, which needs Pain X "
        "(usually via Attack X first) and Province Y.\n"
        "\n"
        "When finished, output on its own line exactly:\n"
        "FINAL PLAN:\n"
        "then ONE action per line, using the SAME natural-language wording as the "
        "example ('attack object a', 'overcome object a from object b', ...), and "
        "close with a final line:\n"
        "[PLAN END]\n"
        "Do not number the actions and do not write anything after [PLAN END]."
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
            # Mas presupuesto: ahora hay razonamiento ademas del plan.
            max_new_tokens=1024,
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
            # enable_thinking=True daria mejor razonamiento pero puede violar el
            # limite de 2 min en Colab; el CoT va en el canal de respuesta.
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
            if accion:
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
