import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import openai
from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI
from langchain.memory import Memory
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, Integer, String, Text, Table, MetaData
import uvicorn

# Setup FastAPI app
app = FastAPI()

# Setup database
DATABASE_URL = "postgresql+asyncpg://user00:pass@localhost:5432/market_research"
engine = create_async_engine(DATABASE_URL, echo=True)
SessionLocal = sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
metadata = MetaData()

competitor_table = Table(
    "competitor_data",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("query", String, nullable=False),
    Column("data", Text, nullable=False),
)

class CompetitorData(BaseModel):
    query: str
    data: str

# Initialize OpenAI API
openai.api_key = "your_openai_api_key"

# LangChain tools and agent initialization
def initialize_market_research_agent():
    tools = [
        Tool(
            name="Data Collector",
            func=lambda query: f"Mock data collected for {query}",
            description="Collects data from mock sources"
        )
    ]
    llm = OpenAI(api_key="your_openai_api_key")
    memory = Memory()
    return initialize_agent(tools, llm, memory=memory)

agent = initialize_market_research_agent()

@app.on_event("startup")
async def startup():
    async with engine.begin() as conn:
        await conn.run_sync(metadata.create_all)

@app.on_event("shutdown")
async def shutdown():
    await engine.dispose()

@app.post("/analyze")
async def analyze_competitor(data: CompetitorData):
    try:
        # Decompose query and generate insights
        sub_tasks = agent.decompose(data.query)
        insights = [agent.run(task) for task in sub_tasks]
        final_result = "\n".join(insights)

        # Save to database
        async with SessionLocal() as session:
            async with session.begin():
                query_result = competitor_table.insert().values(query=data.query, data=final_result)
                await session.execute(query_result)

        return {"query": data.query, "insights": final_result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/results/{query}")
async def get_results(query: str):
    try:
        async with SessionLocal() as session:
            async with session.begin():
                result = await session.execute(
                    competitor_table.select().where(competitor_table.c.query == query)
                )
                data = result.fetchone()
                if not data:
                    raise HTTPException(status_code=404, detail="Query not found.")
                return {"query": data.query, "insights": data.data}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
