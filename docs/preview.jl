using LiveServer

# Use servedocs which:
# 1. Automatically runs make.jl
# 2. Watches for changes in doc files
# 3. Automatically rebuilds when changes are detected
# 4. Includes watching src directory for docstring changes (with Revise)
servedocs(
    doc_env=true,                    # Use the docs environment
    include_dirs=["../src/"],        # Watch package source code for docstring changes
    launch_browser=false             # Don't open the browser automatically
) 

# Server will start at http://localhost:8000 by default
println("Documentation server started at http://localhost:8000")
println("To view in VS Code, open Simple Browser and navigate to this URL") 