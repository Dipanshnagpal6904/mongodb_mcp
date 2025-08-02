import asyncio
import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from mcp_use import MCPAgent, MCPClient


def make_case_insensitive(obj, schema):
    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            if isinstance(v, str) and k in schema.get("string_fields", []):
                result[k] = {"$regex": v, "$options": "i"}
            elif isinstance(v, dict):
                result[k] = make_case_insensitive(v, schema)
            else:
                result[k] = v
        return result
    return obj

# Suggest top 3 DB/collection pairs
async def suggest_db_collection_candidates(llm, client, user_input):
    try:
        tool = client.get_tool("list-databases")
        dbs = await tool.ainvoke({})
        db_names = [d['name'] for d in dbs['databases']]

        db_coll_map = {}
        for db in db_names:
            ctool = client.get_tool("list-collections")
            colls = await ctool.ainvoke({"database": db})
            db_coll_map[db] = [c['name'] for c in colls['collections']]

        prompt = f"""
User input: "{user_input}"

These are the available databases and collections:
{json.dumps(db_coll_map, indent=2)}

Return a JSON array with the top 3 most relevant DB/collection pairs, like:
[
  {{"database": "db1", "collection": "coll1"}},
  ...
]
"""
        suggestions = llm.invoke(prompt)
        return json.loads(suggestions.content.strip())
    except Exception as e:
        print(f" DB/collection suggestion failed: {e}")
        return [{"database": "car_database", "collection": "cars"}]

# Extract schema and identify string fields
async def get_collection_schema(client, db, coll):
    schema = await client.collection_schema(db, coll)
    string_fields = [f["name"] for f in schema.get("fields", []) if f.get("bsonType") == "string"]
    return {**schema, "string_fields": string_fields}

# Smart $match retry
async def try_query_with_match(llm, client, db, coll, user_input, schema):
    try:
        print(f" Trying smart $match on: {db}.{coll}")
        match_prompt = f"""
User request: "{user_input}"

Here is the collection schema:
{json.dumps(schema, indent=2)}

"""
        response = llm.invoke(match_prompt)
        raw = json.loads(response.content.strip())
        final = make_case_insensitive(raw, schema)
        pipeline = [{"$match": final}]
        result = await client.aggregate(db, coll, pipeline)
        if isinstance(result, str) and "Found 0" in result:
            raise ValueError("No data found")
        return result
    except Exception as e:
        print(f"⚠️ Smart $match failed on {db}.{coll}: {e}")
        return None

# Fallback to LLM-generated aggregation pipeline
async def try_query_with_aggregation(llm, client, db, coll, user_input, schema):
    try:
        print(f"🔁 Trying full aggregation on: {db}.{coll}")
        agg_prompt = f"""
User query: "{user_input}"

Here is the schema:
{json.dumps(schema, indent=2)}

Generate a complete MongoDB aggregation pipeline (JSON array).

Use $match, $group, $sort, $project etc. if needed.

Respond ONLY with a valid JSON array.
"""
        response = llm.invoke(agg_prompt)
        pipeline = json.loads(response.content.strip())
        result = await client.aggregate(db, coll, pipeline)
        return result
    except Exception as e:
        print(f"❌ Aggregation failed: {e}")
        return None

async def main():
    load_dotenv()
    client = MCPClient.from_config_file("mcp.json")
    llm = ChatOpenAI(model="gpt-4o", streaming=False)
    agent = MCPAgent(llm=llm, client=client, max_steps=30)

    print("💬 Smart NLP MongoDB Agent (multi-attempt fallback enabled)\n")

    while True:
        try:
            user_input = input("🧑 You: ").strip()
            if user_input.lower() in {"exit", "quit"}:
                print("👋 Goodbye!")
                break

            print("\n📝 Planning query...")
            plan = llm.invoke(f"Break down this MongoDB task: {user_input}")
            print(f"🪜 Steps:\n{plan.content}\n")

            print("🤖 Executing via MCPAgent...")
            result = await agent.run(user_input)
            if isinstance(result, str) and "Found 0" in result:
                raise ValueError("No documents found")
            print(f"✅ Result:\n{result}\n")
            continue

        except Exception as e:
            print(f"⚠️ Agent fallback triggered: {e}")

        # Multi-attempt retry path
        candidates = await suggest_db_collection_candidates(llm, client, user_input)

        for candidate in candidates:
            db, coll = candidate["database"], candidate["collection"]
            try:
                schema = await get_collection_schema(client, db, coll)
            except Exception as schema_error:
                print(f"❌ Schema load failed for {db}.{coll}: {schema_error}")
                continue

            match_result = await try_query_with_match(llm, client, db, coll, user_input, schema)
            if match_result:
                print(f"✅ Smart Match Result from {db}.{coll}:\n{match_result}\n")
                break

            agg_result = await try_query_with_aggregation(llm, client, db, coll, user_input, schema)
            if agg_result:
                print(f"✅ Aggregation Result from {db}.{coll}:\n{agg_result}\n")
                break
        else:
            print("❌ All retries failed. No data found.")

if __name__ == "__main__":
    asyncio.run(main())