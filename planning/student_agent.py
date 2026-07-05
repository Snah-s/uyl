import re


class AssemblyAgent:
    """
    Agente de planning para Qwen3-8B.

    Idea central
    ------------
    Cada `scenario_context` YA es un prompt one-shot: contiene las reglas del
    dominio + UN ejemplo resuelto (en lenguaje natural) + el problema real cuyo
    ultimo bloque [PLAN] esta vacio. Por tanto no reformulamos el prompt: dejamos
    que el modelo CONTINUE ese ultimo [PLAN] imitando el estilo del ejemplo y
    despues convertimos cada linea en lenguaje natural al formato canonico que
    espera el evaluador:  (verbo arg1 [arg2]).

    Dominios
    --------
    - mystery/objects : attack/succumb (1 arg), feast/overcome (2 args)
    - blocks          : engage_payload/release_payload (1), mount_node/unmount_node (2)
    """

    SYSTEM_PROMPT = (
        "Eres un planificador logico experto. Continua UNICAMENTE el ultimo [PLAN] "
        "vacio del problema final. Escribe una accion por linea, con EXACTAMENTE el "
        "mismo estilo y vocabulario que el plan del ejemplo anterior. No repitas el "
        "enunciado ni el ejemplo, no numeres, no anadas explicaciones. Termina con "
        "[PLAN END]."
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
            max_new_tokens=512,
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
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
    # Parseo: texto del modelo -> lista de acciones canonicas
    # ------------------------------------------------------------------ #
    def _parsear_plan(self, texto: str, dominio: str) -> list:
        acciones = []
        for linea in texto.splitlines():
            l = linea.strip()
            if not l:
                continue
            low = l.lower()

            # Cortamos en cuanto el modelo cierra el plan o abre otro problema.
            if "[plan end]" in low or "[statement]" in low or "[plan]" in low:
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
