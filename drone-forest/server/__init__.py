"""Drone-forest decision microservice.

The game (web/) streams sensor frames over a WebSocket; this service answers each one with a
flight action chosen by whichever `DecisionEngine` is selected. Engines are registered in
`server.engines` and selected by name, so a new engine is one new module and one decorator.
"""
