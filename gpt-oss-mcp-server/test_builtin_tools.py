from openai import OpenAI
import json
from pprint import pprint
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.syntax import Syntax
from rich import box
from dotenv import load_dotenv

load_dotenv()


def display_response_flow(response):
    """
    Display the response in a nice, structured format showing:
    - Each reasoning step
    - Each tool call with its action
    - The final assistant message
    - Token usage statistics
    """
    console = Console()
    
    # Header with status
    status_color = {
        'completed': 'green',
        'in_progress': 'yellow',
        'failed': 'red',
        'cancelled': 'red'
    }.get(response.status, 'cyan')
    
    console.print()
    console.print(Panel.fit(
        f"[bold cyan]Response Flow: {response.model}[/bold cyan]\n"
        f"[bold]Status:[/bold] [{status_color}]{response.status}[/{status_color}]",
        border_style="cyan"
    ))
    console.print()
    
    if not hasattr(response, 'output') or not response.output:
        console.print("[yellow]No output in response[/yellow]")
        return
    
    step_num = 1
    reasoning_count = 0
    tool_call_count = 0
    has_final_message = False
    
    for output_item in response.output:
        # Display reasoning blocks
        if output_item.type == 'reasoning':
            reasoning_count += 1
            for content in output_item.content:
                if content.type == 'reasoning_text':
                    console.print(Panel(
                        Text(content.text, style="italic dim"),
                        title=f"[bold yellow]💭 Reasoning #{reasoning_count}[/bold yellow]",
                        border_style="yellow",
                        box=box.ROUNDED,
                        padding=(1, 2)
                    ))
                    console.print()
        
        # Display tool calls
        elif output_item.type == 'web_search_call':
            tool_call_count += 1
            action = output_item.action
            status = output_item.status or "unknown"
            
            # Format action details - action is a Pydantic model, not a dict
            action_type = getattr(action, 'type', 'N/A')
            action_text = f"[bold]Type:[/bold] {action_type}\n"
            
            if hasattr(action, 'query') and action.query:
                action_text += f"[bold]Query:[/bold] {action.query}\n"
            if hasattr(action, 'url') and action.url:
                action_text += f"[bold]URL:[/bold] {action.url}\n"
            if hasattr(action, 'pattern') and action.pattern:
                action_text += f"[bold]Pattern:[/bold] {action.pattern}\n"
            
            action_text += f"[bold]Status:[/bold] {status}"
            
            # Choose icon based on action type
            icon = {
                'search': '🔍',
                'open_page': '📄',
                'find': '🔎',
            }.get(action_type, '🔧')
            
            console.print(Panel(
                action_text,
                title=f"[bold green]{icon} Tool Call #{tool_call_count}[/bold green]",
                border_style="green",
                box=box.ROUNDED,
                padding=(1, 2)
            ))
            console.print()
        
        # Display final message
        elif output_item.type == 'message' and output_item.role == 'assistant':
            has_final_message = True
            for content in output_item.content:
                if content.type == 'output_text':
                    console.print(Panel(
                        Text(content.text),
                        title="[bold blue]📨 Final Response[/bold blue]",
                        border_style="blue",
                        box=box.DOUBLE,
                        padding=(1, 2)
                    ))
                    console.print()
    
    # Warning if no final message
    if not has_final_message:
        console.print(Panel(
            "[bold yellow]⚠️  No final response message found![/bold yellow]\n"
            "The model may have been cut off or encountered an error.",
            border_style="yellow",
            box=box.ROUNDED
        ))
        console.print()
    
    # Usage statistics
    if hasattr(response, 'usage') and response.usage:
        usage = response.usage
        
        stats_text = f"""[bold]Input tokens:[/bold] {usage.input_tokens:,}
[bold]Output tokens:[/bold] {usage.output_tokens:,}
[bold]Total tokens:[/bold] {usage.total_tokens:,}"""
        
        if hasattr(usage, 'input_tokens_details') and usage.input_tokens_details:
            if hasattr(usage.input_tokens_details, 'cached_tokens'):
                cached = usage.input_tokens_details.cached_tokens
                stats_text += f"\n[bold]Cached tokens:[/bold] {cached:,} ({cached/usage.input_tokens*100:.1f}%)"
        
        if hasattr(usage, 'output_tokens_details') and usage.output_tokens_details:
            details = usage.output_tokens_details
            if hasattr(details, 'reasoning_tokens'):
                stats_text += f"\n[bold]Reasoning tokens:[/bold] {details.reasoning_tokens:,}"
            if hasattr(details, 'tool_output_tokens'):
                stats_text += f"\n[bold]Tool output tokens:[/bold] {details.tool_output_tokens:,}"
        
        console.print(Panel(
            stats_text,
            title="[bold magenta]📊 Token Usage[/bold magenta]",
            border_style="magenta",
            box=box.ROUNDED,
            padding=(1, 2)
        ))
        console.print()
    
    # Summary
    console.print(Panel.fit(
        f"[bold]Summary:[/bold] {reasoning_count} reasoning steps, {tool_call_count} tool calls",
        border_style="cyan"
    ))
    console.print()


client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)
 
response = client.responses.create(
    model="openai/gpt-oss-20b",
    input="How is the weather in Seattle, WA?",
    tools=[
        {
            "type": "code_interpreter",
            "container": {
                "type": "auto"
            }
        },
        {
            "type": "web_search_preview"
        }
    ],
    reasoning={
        "effort": "low", # "low", "medium", or "high"
        "summary": "detailed"  # "auto", "concise", or "detailed"
    },
    temperature=1.0,
    max_output_tokens=4096
)

# Display the response in a nice format
display_response_flow(response)

# Show raw response for debugging if response looks incomplete
show_raw_debug = response.status != 'completed'
if show_raw_debug:  # Set to False to hide raw response
    print("\n" + "=" * 80)
    print("FULL RAW RESPONSE (for debugging)")
    print("=" * 80)
    response_dict = response.model_dump() if hasattr(response, 'model_dump') else dict(response)
    pprint(response_dict, width=120, depth=10) 