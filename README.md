# EDAgent

A multi-agent system for generating and verifying OpenROAD Python scripts using RAG and multiple LLMs.

## Setup

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Copy `env.example` to `.env` and fill in your API keys:
```bash
cp env.example .env
```
4. Set up Milvus:
```bash
# Start Docker if not running
# Then navigate to milvus directory and start the container
cd milvus
docker-compose up -d

# Return to project root and run setup
cd ..
python setup_milvus.py
```

## Usage

1. First, preprocess the data:
```bash
python read_data.py
```

2. Run the benchmarks:
```bash
# Run non-RAG benchmark
python benchmark_non_rag.py

# Run RAG benchmark
python benchmark_rag.py

# Run multi-agent workflow
python agent_workflow.py
```

## Agent Architecture

The system uses a multi-agent workflow with the following components:

1. **Drafting Agents**
   - Gemini Drafter: Uses Google's Gemini model to generate initial scripts
   - OpenAI Drafter: Uses OpenAI's GPT model to generate alternative scripts

2. **Consolidation Agent**
   - Takes both drafts and merges them into a single, improved script
   - Uses Gemini model for consolidation

3. **Verification Agent**
   - Checks the consolidated script against OpenROAD API requirements
   - Uses RAG to compare against known good examples
   - Returns VALID or INVALID with required corrections

4. **Correction Agent**
   - Makes necessary corrections based on verification feedback
   - Uses Gemini model for corrections

The workflow follows this sequence:
1. Both drafting agents generate scripts in parallel
2. Consolidation agent merges the drafts
3. Verification agent checks the script
4. If invalid, correction agent fixes issues
5. Process repeats up to 3 times if needed
6. Final script is saved with iteration count

Results are saved to `results/multi_agent_results.csv` with:
- Original prompt
- Final script
- Number of verification iterations

## Results

Results from each benchmark are saved in the `results` directory:
- `RAG_non-thinking.csv`: Results from RAG benchmark
- `non_RAG_non-thinking.csv`: Results from non-RAG benchmark
- `multi_agent_results.csv`: Results from multi-agent workflow
- (we have also included a "reproduced" directory containing our own reproduction of these initial results)

## Validation in OpenROAD

To validate the results scripts generated from the agent we have to run the OpenROAD flow script to run and test the scripts.
1. **Procedure:**
- Inside ORFS flow directory create a new script called flow_script.py and paste the generated script.
- Update the script manually with the appropriate paths for lib dir, design dir, lef dir and tech lef dir. It is adviced to use "nangate45" as your target PDK and "gcd" as your target design. Script expects a synthesized netlist so you'll have to synthesize gcd using the same "nangate45" pdk.
- use command openroad -python -exit flow_script.py to run the script and validate. 

Currently we're using EDA corpus v1 as our current RAG database. EDA corpus v1 is an old version and hence some of the APIs have been deprecated and changed in the newer ORFS versions, also there are few errors in the script generated because LLM hallucination hence you might run into some errors while running the flow. Some of the possible errors have been listed.

2. **Possible Errors:**
- """" quote error: there is an extra quote in some of the API string calls which should be changed to "" .
- metal layer names "M1" : "nangate45" pdk uses "metal1", "metal2" as the naming convention so change that manually.
- Run attribute error for the pin placer : pin placer doesn't have any ".run()" method. Use tcl alternative of io placer command instead with -random flag if necessary
- gpl.doInitialPlace(), gpl.doNesterovPlace() these APIs require no. of threads as an argument. Pass a reasonable integer.
- provide "verilogFile name" and the appropriate "site name" as an input.
  
## Next Steps (before project presentation)
- RAG: Integrate EDA-Corpus v2 (improved/larger dataset, different formats) (Flow-v2.xlsx)
- RAG: Integrate OpenROAD APIs (RAGAPIs.csv)
- RAG: Integrate OpenROAD sample code pieces (RAGCodePiece.csv)
