class ChatBoxError(Exception):
    """Base exception for ChatBox domain errors."""


class IngestionError(ChatBoxError):
    pass


class ParsingError(IngestionError):
    pass


class ChunkingError(IngestionError):
    pass


class RetrievalError(ChatBoxError):
    pass


class GenerationError(ChatBoxError):
    pass


class StorageError(ChatBoxError):
    pass
