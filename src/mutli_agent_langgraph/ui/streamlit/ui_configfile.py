from configparser import ConfigParser

class Config:
    def __init__(self, config_file='src/mutli_agent_langgraph/ui/streamlit/uiconfig.ini'):
        self.config = ConfigParser()
        self.config.read(config_file)

    def get_title(self):
        return self.config["DEFAULT"].get("TTITLE", fallback="QEA Cognitive Assistant")
    
    def get_subtitle(self):
        return self.config["DEFAULT"].get("SUBTITLE", fallback="A Multi-Agent LangGraph Application")
    
    def get_llm_options(self):
        return self.config["DEFAULT"].get("LLM_OPTIONS").split(", ")
    
    def get_usecase_options(self):
        return self.config["DEFAULT"].get("USECASE_OPTIONS").split(", ")
    
    def get_openai_model(self):
        return self.config["DEFAULT"].get("OPENAI_Model")
    
    def get_groq_model(self):
        return self.config["DEFAULT"].get("GROQ_Model").split(", ")
    
    def get_google_model(self):
        return self.config["DEFAULT"].get("GOOGLE_Model").split(", ")
    
    def get_ollama_model(self):
        return self.config["DEFAULT"].get("OLLAMA_Model").split(", ")
    
    def get_api_key(self):
        return self.config["DEFAULT"].get("API_KEY").split(", ")
    
    def get_language(self):
        return self.config["DEFAULT"].get("LANGUAGE").split(", ")
    
    def get_testing_framework(self):
        return self.config["DEFAULT"].get("FRAMEWORKS").split(", ")
    
    def get(self, section: str, option: str, fallback=None):
        return self.config.get(section, option, fallback=fallback)

    def getboolean(self, section: str, option: str, fallback=None):
        return self.config.getboolean(section, option, fallback=fallback)

    def getint(self, section: str, option: str, fallback=None):
        return self.config.getint(section, option, fallback=fallback)

    def tracking(self, key: str, fallback=None, as_bool=False, as_int=False):
        """
        Shortcut to access [tracking] section values.

        Example:
            cfg.tracking("enable_mlflow", as_bool=True)
            cfg.tracking("mlflow_tracking_uri")
        """
        if as_bool:
            return self.getboolean("tracking", key, fallback=fallback)
        elif as_int:
            return self.getint("tracking", key, fallback=fallback)
        else:
            return self.get("tracking", key, fallback=fallback)
