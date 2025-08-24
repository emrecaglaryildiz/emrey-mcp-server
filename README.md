*-*-*-*-* EMREY MCP SERVER *-*-*-*-*
------------------------------------


+ #First, install the necessary packages.
- pip install -r requirements.txt

+ Then, start the server and check for any errors. If there are any errors or missing packages, check them.
- python mcp-server.py

+ If no issues are observed, you can now define the necessary path settings for the mcp.json file from the LM Studio menu and start using it.
-
{
  "mcpServers": {
    "webtools": {
      "command": "python.exe",
      "args": [
        "C:/mcp/server.py"
      ]
    }
  }
}

-
PS NOTE: The MCP server has the following features, and you can add as many as you want. :)

- web_search
- http_get
- http_get_chunks
- text_to_pdf
