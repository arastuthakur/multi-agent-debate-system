import os
import asyncio
from typing import List, Dict, Any
from dotenv import load_dotenv
from groq import Groq
import google.generativeai as genai
import sys
import time
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress

# Initialize Rich console for better output
console = Console()

# Load environment variables
load_dotenv()

class LLMClient:
    def __init__(self, provider: str):
        self.provider = provider
        try:
            if provider == "groq":
                api_key = os.getenv("GROQ_API_KEY")
                if not api_key:
                    raise ValueError("GROQ_API_KEY not found in .env file")
                self.client = Groq(api_key=api_key)
                self.model = "deepseek-r1-distill-llama-70b"
            elif provider == "gemini":
                api_key = os.getenv("GOOGLE_API_KEY")
                if not api_key:
                    raise ValueError("GOOGLE_API_KEY not found in .env file")
                genai.configure(api_key=api_key)
                self.client = genai.GenerativeModel('gemini-pro')
            else:
                raise ValueError(f"Unsupported provider: {provider}")
        except Exception as e:
            console.print(f"[red]Error initializing {provider} client: {str(e)}[/red]")
            sys.exit(1)

    def generate_stream(self, messages: List[Dict[str, str]], **kwargs) -> str:
        try:
            if self.provider == "groq":
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=kwargs.get('temperature', 0.7),
                    max_completion_tokens=4096,
                    top_p=0.95,
                    stream=True,
                    stop=None,
                )
                for chunk in completion:
                    yield chunk.choices[0].delta.content or ""
            else:  # gemini
                prompt = "\n".join(f"{msg['role']}: {msg['content']}" for msg in messages)
                response = self.client.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=kwargs.get('temperature', 0.7)
                    ),
                    stream=True
                )
                for chunk in response:
                    if chunk.text:
                        yield chunk.text
        except Exception as e:
            console.print(f"[red]Error generating response: {str(e)}[/red]")
            yield "Error generating response. Please try again."

class Agent:
    def __init__(self, name: str, llm_client: LLMClient, role: str):
        self.name = name
        self.llm_client = llm_client
        self.role = role

    async def generate_response(self, context: Dict[str, Any]) -> str:
        messages = [
            {"role": "system", "content": self.role},
            {"role": "user", "content": context.get("question", "")}
        ]
        
        if "discussion" in context:
            for entry in context["discussion"]:
                messages.append({
                    "role": "assistant",
                    "content": f"{entry['agent']}: {entry['response']}"
                })
        
        response = ""
        with Progress() as progress:
            task = progress.add_task(f"[cyan]{self.name} is thinking...", total=None)
            for chunk in self.llm_client.generate_stream(messages):
                response += chunk
                progress.update(task, advance=1)
        return response

class DebateSystem:
    def __init__(self):
        console.print(Panel.fit("[yellow]Initializing Multi-Agent Debate System[/yellow]"))
        
        try:
            groq_client = LLMClient("groq")
            gemini_client = LLMClient("gemini")
            
            self.agents = {
                "proposer": Agent(
                    "Proposer",
                    groq_client,
                    "You are an agent that proposes initial solutions. Be creative and thorough in your answers."
                ),
                "critic": Agent(
                    "Critic",
                    gemini_client,
                    "You are a critical thinker. Analyze the previous response and point out potential flaws, missing considerations, or areas of improvement."
                ),
                "refiner": Agent(
                    "Refiner",
                    groq_client,
                    "You are a solution refiner. Take the original proposal and criticisms into account to provide an improved solution."
                ),
                "synthesizer": Agent(
                    "Synthesizer",
                    gemini_client,
                    "You are a final synthesizer. Analyze the entire discussion and provide a well-reasoned final answer that incorporates the best ideas while addressing the criticisms."
                )
            }
            
            self.discussion_history: List[Dict[str, str]] = []
            console.print("[green]✓ System initialized successfully![/green]")
        except Exception as e:
            console.print(f"[red]Error initializing debate system: {str(e)}[/red]")
            sys.exit(1)

    async def debate(self, question: str, max_iterations: int = 2) -> str:
        context = {
            "question": question,
            "discussion": self.discussion_history
        }

        try:
            # 1. Get initial proposal
            console.print("\n[cyan]Step 1: Getting initial proposal[/cyan]")
            proposal = await self.agents["proposer"].generate_response(context)
            self.discussion_history.append({"agent": "Proposer", "response": proposal})
            console.print(Panel(proposal, title="[yellow]Proposer[/yellow]", border_style="yellow"))

            for i in range(max_iterations):
                console.print(f"\n[cyan]Iteration {i+1}/{max_iterations}[/cyan]")
                
                # 2. Get criticism
                context["discussion"] = self.discussion_history
                criticism = await self.agents["critic"].generate_response(context)
                self.discussion_history.append({"agent": "Critic", "response": criticism})
                console.print(Panel(criticism, title="[red]Critic[/red]", border_style="red"))

                # 3. Get refined solution
                context["discussion"] = self.discussion_history
                refinement = await self.agents["refiner"].generate_response(context)
                self.discussion_history.append({"agent": "Refiner", "response": refinement})
                console.print(Panel(refinement, title="[blue]Refiner[/blue]", border_style="blue"))

            # 4. Get final synthesis
            console.print("\n[cyan]Final Step: Synthesizing discussion[/cyan]")
            context["discussion"] = self.discussion_history
            final_answer = await self.agents["synthesizer"].generate_response(context)
            self.discussion_history.append({"agent": "Synthesizer", "response": final_answer})
            console.print(Panel(final_answer, title="[green]Final Synthesis[/green]", border_style="green"))

            return final_answer
            
        except Exception as e:
            console.print(f"[red]Error during debate: {str(e)}[/red]")
            return "An error occurred during the debate. Please try again."

async def main():
    # Clear screen for better presentation
    os.system('cls' if os.name == 'nt' else 'clear')
    
    console.print(Panel.fit("""
[yellow]Multi-Agent Debate System[/yellow]
This system uses multiple AI agents to analyze and discuss your question:
- [yellow]Proposer[/yellow]: Generates initial solutions
- [red]Critic[/red]: Points out potential flaws
- [blue]Refiner[/blue]: Improves the solution
- [green]Synthesizer[/green]: Creates final synthesis
    """))

    debate_system = DebateSystem()
    
    while True:
        try:
            question = console.input("\n[cyan]Enter your question (or 'quit' to exit): [/cyan]")
            
            if question.lower() in ['quit', 'exit', 'q']:
                console.print("[yellow]Thank you for using the Multi-Agent Debate System![/yellow]")
                break
                
            if not question.strip():
                console.print("[red]Please enter a valid question.[/red]")
                continue

            console.print(Panel(question, title="[cyan]Question[/cyan]", border_style="cyan"))
            
            start_time = time.time()
            final_answer = await debate_system.debate(question)
            end_time = time.time()
            
            console.print(f"\n[cyan]Debate completed in {end_time - start_time:.2f} seconds[/cyan]")
            console.print("\n[yellow]Would you like to ask another question?[/yellow]")
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Gracefully shutting down...[/yellow]")
            break
        except Exception as e:
            console.print(f"[red]An error occurred: {str(e)}[/red]")
            console.print("[yellow]Would you like to try again?[/yellow]")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Goodbye![/yellow]")
    except Exception as e:
        console.print(f"[red]Fatal error: {str(e)}[/red]")
