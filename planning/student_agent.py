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

    * v3 (salida directa + reglas de precondicion duras + dedup + 256 tokens):
      **1.25/10, 111s**. Arreglado el tiempo (~3.7s/caso, bajo el limite) y el
      score duplica a v1. Aparecen 2 aciertos perfectos (10/10). Nuevo fallo
      dominante: en idx-1, tras un `feast` correcto, el modelo hace `overcome`
      cuando el optimo hace `succumb`. Ademas mi instruccion "as SHORT as
      possible" producia planes DEMASIADO cortos (perdian el +2 de longitud).

    Estrategia v4 (esta)
    --------------------
    Regla de orden EXACTA para el idx-1: Attack y Feast CONSUMEN Harmony;
    Overcome NO necesita Harmony. Por tanto, tras un feast/attack, si el siguiente
    paso es otro attack/feast hay que `succumb` primero; si es overcome, no.
    Ademas: apertura por craves (hay craves iniciales -> Feast; no hay -> Attack),
    y se quita "as SHORT as possible" (pedia planes cortos de mas) por "alcanza la
    meta completa; los planes suelen ser de 2, 4 o 6 acciones".

    Estrategia v5 (esta)
    --------------------
    Sobre v4 se anade DETECCION DE CICLOS en el parser. Diagnostico sobre Task:
    36/50 planes salian con longitud impar o >6 (el optimo SIEMPRE es 2, 4 o 6),
    con casos de 8/10/12/16 acciones = el modelo repitiendo bloques de 2-3
    acciones. El dedup consecutivo (v3) solo mata A A; ahora `_romper_ciclos`
    corta A B A B / A B C A B C, recuperando esos planes degenerados sin tocar
    los sanos.

    Dominios
    --------
    - mystery/objects : attack/succumb (1 arg), feast/overcome (2 args)
    - blocks          : engage_payload/release_payload (1), mount_node/unmount_node (2)
    """

    # Marcador opcional: si el modelo lo emitiese, el parser toma solo lo de despues.
    FINAL_MARKER = "final plan:"

    SYSTEM_PROMPT_MYSTERY = (
        "You are an expert STRIPS planner. The problem gives the domain rules, ONE "
        "solved example, and a final problem whose last [PLAN] is empty. Output "
        "ONLY the plan for the FINAL problem: one action per line, using the SAME "
        "wording as the example, then a final line [PLAN END]. No explanations, no "
        "numbering, no thinking, and never repeat an action twice in a row.\n"
        "Reach the COMPLETE goal (every required craving); do NOT stop early. "
        "Optimal plans are usually 2, 4 or 6 actions long.\n"
        "\n"
        "Mystery actions and how they constrain the ORDER:\n"
        "- Attack X needs Province X + Planet X + Harmony; produces Pain X and "
        "REMOVES Harmony.\n"
        "- Feast X from Y needs 'X craves Y' + Province X + Harmony; produces "
        "Pain X + Province Y and REMOVES Harmony and the craving.\n"
        "- Succumb X needs Pain X; restores Province X + Planet X + Harmony.\n"
        "- Overcome X from Y needs Pain X + Province Y (it does NOT need Harmony); "
        "produces 'X craves Y' + Province X + Harmony.\n"
        "\n"
        "Ordering rules (follow strictly):\n"
        "1. FIRST action: if the initial conditions list any 'X craves Y', begin "
        "with a Feast; if there are no cravings initially, begin with an Attack. "
        "NEVER begin with Overcome or Succumb (there is no Pain yet).\n"
        "2. Attack and Feast both CONSUME Harmony; Overcome does NOT need Harmony. "
        "So right after an Attack or a Feast: if the next step is another Attack or "
        "Feast, you MUST insert 'succumb' first to get Harmony back; if the next "
        "step is an Overcome, do NOT succumb.\n"
        "3. To create 'X craves Y', finish that part with 'overcome X from Y' "
        "(needs Pain X + Province Y)."
    )

    SYSTEM_PROMPT_BLOCKS = (
        "You are an expert blocksworld planner. The problem gives the domain rules, "
        "ONE solved example, and a final problem whose last [PLAN] is empty. Output "
        "ONLY the plan for the FINAL problem: one action per line, using EXACTLY the "
        "same wording as the example ('pick up the red block', 'unmount_node the "
        "blue block from on top of the red block', 'put down the blue block', "
        "'mount_node the red block on top of the orange block'), then a final line "
        "[PLAN END]. No explanations, no numbering, no thinking.\n"
        "Constraints: the hand holds at most one block; you may pick up or "
        "unmount_node a block only if your hand is empty and that block is "
        "unobstructed (no block on top); to mount_node a block onto another you must "
        "be holding it and the target block must be unobstructed; you may only pick "
        "up a block that is on the table.\n"
        "\n"
        "Strategy (follow strictly):\n"
        "1. First UNMOUNT_NODE every block that is on top of a wrong block and PUT "
        "it DOWN, working from the topmost block downward, until every block you "
        "need is free and on the table.\n"
        "2. Then build each goal tower from the BOTTOM up: pick up the lower block, "
        "mount_node the next block on top of it, and repeat upward.\n"
        "3. Alternate correctly: after pick up / unmount_node your hand is FULL, so "
        "the next action must be put down / mount_node; never pick up twice in a "
        "row. Stop as soon as the goal stacking holds."
    )

    def __init__(self):
        # Compatibilidad: expuesto pero solve() elige el prompt por dominio.
        self.system_prompt = self.SYSTEM_PROMPT_MYSTERY

    # ------------------------------------------------------------------ #
    # API principal
    # ------------------------------------------------------------------ #
    def solve(self, scenario_context: str, llm_engine_func) -> list:
        # El system depende del dominio: las reglas de mystery (Harmony/Province/
        # craves) son ruido daniño para blocks, y viceversa.
        dominio = self._detectar_dominio(scenario_context)
        system = (
            self.SYSTEM_PROMPT_BLOCKS if dominio == "blocks"
            else self.SYSTEM_PROMPT_MYSTERY
        )

        respuesta = llm_engine_func(
            prompt=scenario_context,
            system=system,
            # Plan corto (<=6 acciones ~ 6 lineas). Un tope bajo mantiene ~4s/caso
            # y CORTA los bucles degenerados que aparecian con 1024 tokens.
            max_new_tokens=256,
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
            # thinking desactivado a proposito: dispara el tiempo (>2 min).
            enable_thinking=False,
        )

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

        # Segunda red: cortar BUCLES (no solo duplicados consecutivos). En Task
        # el modelo degenera en Ciclos de 2-3 acciones que se repiten
        # (planes de 8/10/12/16 acciones, imposibles: el optimo es 2, 4 o 6).
        return self._romper_ciclos(acciones)

    @staticmethod
    def _romper_ciclos(acciones: list) -> list:
        """Trunca en cuanto el plan empieza a repetir un bloque de 1-3 acciones.

        En mystery/blocks la precondicion de una accion desaparece al ejecutarla,
        asi que un bloque adyacente repetido (A B A B, A B C A B C, ...) es SIEMPRE
        degeneracion del modelo, no plan valido. Al detectar el primer ciclo
        conservamos una sola copia del bloque y descartamos toda la cola posterior
        (que ya es basura). No toca planes sanos (<=6 acciones sin repeticion).
        """
        out = []
        for a in acciones:
            out.append(a)
            n = len(out)
            for k in (1, 2, 3):
                if n >= 2 * k and out[-2 * k:-k] == out[-k:]:
                    del out[-k:]      # quita la repeticion; corta el bucle aqui
                    return out
        return out

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
        # unmount_node/unstack ANTES que mount_node/stack (subcadenas entre si).
        # Aceptamos el alias 'unstack'/'stack' por si el modelo copia la cabecera
        # del dominio ("Stack a block ...") en vez del ejemplo ("mount_node ...").
        m = re.search(
            r"(?:unmount_node|unstack)\s+the\s+(\w+)\s+block\s+from\s+on\s+top\s+of\s+the\s+(\w+)\s+block",
            low,
        )
        if m:
            return f"(unmount_node {m.group(1)} {m.group(2)})"
        m = re.search(
            r"(?:mount_node|stack)\s+the\s+(\w+)\s+block\s+on\s+top\s+of\s+the\s+(\w+)\s+block",
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
