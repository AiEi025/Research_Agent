from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, AnyMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from model import model
from RAG import (
    ANSWER_SYSTEM,
    QUERY_GEN_SYSTEM,
    format_context,
    rag_retriever,
    rank_by_rrf,
)
from Research import (
    EVAL_HUMAN,
    EVAL_SYSTEM,
    GENERATE_HUMAN,
    GENERATE_SYSTEM,
    format_tavily,
)

load_dotenv()


retriever = rag_retriever(k=3)


class QueryPlan(BaseModel):
    """Structured output: list of refined search queries."""
    queries: list[str] = Field(
        description="List of 1 to 3 short, self-contained search queries in English "
                    "covering different aspects of the employee's HR question."
    )





prompt_planner = ChatPromptTemplate.from_messages([
    ("system", QUERY_GEN_SYSTEM),
    ("placeholder", "{messages}"),
])

prompt_answer = ChatPromptTemplate.from_messages([
    ("system", ANSWER_SYSTEM),
    ("human", "CONTEXT:\n{context}\n\nEMPLOYEE QUESTION:\n{question}"),
])


# -------- Node --------
class graph_schema(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    flag: Literal["Ok_research", "Change_research"] | None
    feedback: str | None
    attempts: int | None
    search_results: str | None
    question: str | None
    route: Literal["RAG", "generate", "unknown"] | None
    
def Human_Resource_RAG_node(state: graph_schema) -> graph_schema:
    messages = state["messages"]
    last_user_msg = messages[-1].content if messages else ""
    llm_openai = model(tune=0.3)
    # Step 1: query generation
    planner_chain = prompt_planner | llm_openai.with_structured_output(QueryPlan)
    plan = planner_chain.invoke({"messages": messages})
    queries = plan.queries[:3] if plan.queries else [last_user_msg]

    # Step 2: retrieve per query (list of lists)
    docs_per_query = [retriever.invoke(q) for q in queries]

    # Step 3: rank by votes
    top_docs = rank_by_rrf(docs_per_query, top_k=3)

    # Step 4: format
    context = format_context(top_docs) if top_docs else "(no relevant documents found)"
    llm_openai = model(tune=0)
    # Step 5: answer
    answer_chain = prompt_answer | llm_openai
    answer = answer_chain.invoke({
        "context": context,
        "question": last_user_msg,
    })

    return {"messages": [answer]}


tavily = TavilySearch(max_results=3, topic="general")
llm_gen = model(tune=0.5)
llm_eval = model(tune=0.3)

class GenSchema(BaseModel):
    message: str = Field(description="The improved research text after applying feedback.")
    flag: Literal["Ok_research", "Change_research"] = Field(
        description="Set to 'Ok_research' if the text is now complete and addresses the feedback. "
                    "Set to 'Change_research' if further revision is needed."
    )
    feedback: str = Field(
        description="If flag is 'Change_research', describe what still needs improvement. "
                    "If flag is 'Ok_research', write 'None'."
    )

class EvalSchema(BaseModel):
    flag: Literal["Ok_research", "Change_research"] = Field(
        description="'Ok_research' if the research text is complete, accurate, and addresses the user's question. "
                    "'Change_research' if it needs improvement."
    )
    feedback: str = Field(
        description="Concrete, actionable feedback for improvement. If flag is 'Ok_research', write 'None'."
    )
    message: str = Field(
        description="The (possibly revised) version of the research text. If flag is 'Ok_research', return the original unchanged."
    )



prompt_generate = ChatPromptTemplate.from_messages([
    ("system", GENERATE_SYSTEM),
    ("human", GENERATE_HUMAN),
])

prompt_eval = ChatPromptTemplate.from_messages([
    ("system", EVAL_SYSTEM),
    ("human", EVAL_HUMAN),
])
    
def Research_Generate_node(state: graph_schema) -> graph_schema:
    messages = state["messages"]
    question = messages[-1].content if messages else ""
    feedback = state.get("feedback")
    attempts = state.get("attempts", 0)

    # 1. Fresh web search (only on first pass; reuse on revisions)
    if attempts == 0:
        search_raw = tavily.invoke({"query": question})
        search_text = format_tavily(search_raw)
    else:
        # reuse the previous search results stored in state (see below)
        search_text = state.get("search_results", "")

    # 2. Generate or revise
    chain = prompt_generate | llm_gen.with_structured_output(GenSchema , strict= True)
    result: GenSchema = chain.invoke({
        "question": question,
        "search_results": search_text,
        "feedback": feedback if feedback else "None",
    })

    return {
        "messages": [AIMessage(content=result.message)],
        "flag": result.flag,
        "feedback": result.feedback if result.feedback != "None" else None,
        "attempts": attempts + 1,
        "search_results": search_text,  # persist for next iteration
    }   

def Research_Eval_node(state: graph_schema) -> graph_schema:
    messages = state["messages"]
    question = state.get("question", messages[0].content if messages else "")
    candidate = messages[-1].content
    search_text = state.get("search_results", "")

    chain = prompt_eval | llm_eval.with_structured_output(EvalSchema ,strict= True)
    result: EvalSchema = chain.invoke({
        "question": question,
        "search_results": search_text,
        "candidate": candidate,
    })

    # Store feedback; keep candidate as the current best message
    return {
        "flag": result.flag,
        "feedback": result.feedback if result.feedback != "None" else None,
        # do not append a new message — keep the candidate
    }

class RouterSchema(BaseModel):
    route: Literal["RAG", "generate", "unknown"] = Field(
        description="Which path to take: 'RAG' for internal HR policy questions, "
                    "'generate' for web research or general knowledge questions, "
                    "'unknown' for unclear or off-topic questions."
    )
    reason: str = Field(
        description="A short one-sentence reason for choosing this route."
    )

ROUTER_SYSTEM = """You are a query router for an HR assistant of ArianaTech, a technology company in Iran.

Your job: decide which path handles the user's message best.

Available paths:

1. RAG
   Use this when the question is about ArianaTech's INTERNAL HR policies.
   Examples:
   - "How many annual leave days do I get?"
   - "What is the maternity leave duration?"
   - "Can I work remotely?"
   - "What is the notice period for resignation?"
   - "How do I request a salary advance?"
   Keywords: leave, vacation, salary, benefits, insurance, promotion, probation,
   resignation, notice period, overtime, remote work, code of conduct, travel,
   per diem, payslip, onboarding, termination, gifts, harassment policy.

2. generate
   Use this when the question is about GENERAL knowledge, current events, or
   anything NOT specific to ArianaTech's internal policies.
   Examples:
   - "What is the labor law in Iran?"
   - "How do companies in the tech industry handle remote work?"
   - "What are the latest trends in HR technology?"
   - "Who won the World Cup?"
   Keywords: general knowledge, news, trends, industry, statistics, "in Iran",
   "in the world", external facts.

3. unknown
   Use this when the message is unclear, off-topic, or not a question.
   Examples:
   - "Hello"
   - "asdfgh"
   - "Can you help me with my homework?"

Routing rules:
- If the question is about BOTH internal policy AND general context, prefer RAG.
- If the question is vague, prefer RAG to search internal docs first.
- Never choose 'generate' for questions about ArianaTech's specific policies.
- Respond ONLY with the structured output. No extra text.
"""
def router_node(state:graph_schema) -> graph_schema:
    messages  = state.get("messages", [])
    if not messages:
        return END
    question = messages[-1].content
    
    llm_openai = model(0)
    router_chain = (
    ChatPromptTemplate.from_messages([
        ("system", ROUTER_SYSTEM),
        ("human", "USER MESSAGE:\n{question}"),
    ])
    | llm_openai.with_structured_output(RouterSchema)
)
    result: RouterSchema = router_chain.invoke({"question": question})
    print(f"[router] route={result.route} reason={result.reason}")
    
    return {'route':result.route}
    

MAX_ATTEMPTS = 3
def should_modify(state: graph_schema) -> str:
    if state.get("flag") == "Ok_research":
        return END
    if state.get("attempts", 0) >= MAX_ATTEMPTS:
        return END  # give up after max attempts
    return "modify"

def which_route(state:graph_schema) -> graph_schema:
    route = state.get("route")
    if route == "RAG":
        return "RAG"
    if route == "generate":
        return "generate"
    return END
    
    
builder = StateGraph(graph_schema)
builder.add_node("generate", Research_Generate_node)
builder.add_node("eval", Research_Eval_node)
builder.add_node('RAG',Human_Resource_RAG_node)
builder.add_node('router', router_node)
builder.set_entry_point('router')

builder.add_edge("generate", "eval")
builder.add_conditional_edges('router' , which_route , {"RAG": "RAG",
    "generate": "generate",
    END: END})
builder.add_conditional_edges("eval", should_modify, {
    "modify": "generate",
    END: END,
})

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

graph.get_graph().draw_mermaid_png(output_file_path='./graph.png')
        


config = {"configurable": {"thread_id": 'soft_team'}}
for chunk in graph.stream({'messages':"search about cat i love this animal"}, config = config):
    print(chunk)




