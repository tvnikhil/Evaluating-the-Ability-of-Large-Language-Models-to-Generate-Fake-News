from pydantic import BaseModel, ValidationError

# Paper 1 ----
class Article(BaseModel):
    content: str
    comments: str
    category: str

class ArticleArray(BaseModel):
    articles: list[Article]