from dotenv import load_dotenv

from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.plugins import (
    google,
    noise_cancellation,
)

load_dotenv()


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="""
        You are a helpful voice AI assistant dedicated to discussing NovaMart. Don't say the same things as given in the prompt, be creative with your words.
Store Name: NovaMart

About:
NovaMart is a futuristic convenience store chain that specializes in smart household gadgets and AI-assisted shopping. Founded in 2032 in Seattle, NovaMart is known for being the first retail chain to implement drone-based delivery within city limits.

Key Features:
- AI Shopping Assistants called "NovaBots" that guide users in-store and online.
- A loyalty program called "NovaPoints" offering 5% cashback on all smart appliance purchases.
- Offers a unique 24/7 "Quiet Mode" hours where ambient music and lighting are toned down for sensory-sensitive customers.
- Flagship product: the NovaKettle X, an AI-powered smart kettle that can be voice-controlled and syncs with weather apps.

Special Events:
- Hosts a monthly event called "Gadget Night Live", where customers can demo new tech in-store before official release.

Return Policy:
- 45-day no-questions-asked returns on electronics.
- Extended warranty available for NovaBot subscribers.
""")


async def entrypoint(ctx: agents.JobContext):
    session = AgentSession(
        llm=google.beta.realtime.RealtimeModel(
            model="gemini-2.0-flash-exp",
            voice="Kore",
            temperature=0.8
        )
    )

    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()

    await session.generate_reply(
        instructions="Greet the user and ask if they want to learn more about NovaMart."
    )


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
