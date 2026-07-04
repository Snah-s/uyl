class AssemblyAgent:
    def __init__(self):
        self.system_prompt = "Eres un planificador experto en ensamblaje automatizado."
        
    def solve(self, scenario_context: str, llm_engine_func) -> list:
        """
        Recibe el texto del escenario y la funcion del motor LLM.
        Debe retornar una lista de strings con las acciones extraidas.
        """
        prompt_final = f"{scenario_context}\n\nAnaliza la situacion y dame el plan:"
        
        respuesta_bruta = llm_engine_func(
            prompt=prompt_final,
            system=self.system_prompt,
            temperature=0.0,
            do_sample=False # Asegurate de esto
        )
        
        lineas = respuesta_bruta.split('\n')
        acciones_parseadas = []
        for l in lineas:
            l_limpia = l.strip()
            if any(verbo in l_limpia for verbo in ["engage", "mount", "release", "dismount"]):
                acciones_parseadas.append(l_limpia)
                
        return acciones_parseadas