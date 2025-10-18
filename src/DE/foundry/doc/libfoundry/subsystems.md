Title: Subsystems

# Project Contexts

 * [class@Foundry.Context]
 * [class@Foundry.Contextual]
 * [class@Foundry.Inhibitor]

## Context

At the root of the object graph in Foundry is the [class@Foundry.Context].
Generally, it represents a single project on the users system.
However, to allow for project-less use-cases a "shared" context may be created.

See [ctor@Foundry.Context.new] to create a new context for a specific project.
See [ctor@Foundry.Context.new_for_user] to create a new context for the user without involving a project such as for simple text editor or documentation services.

## Contextual

For objects that are to be used solely within the context of a [class@Foundry.Context], subclass [class@Foundry.Contextual].
These objects will have their [property@Foundry.Contextual:context] property cleared when the foundry is shutdown.

Contextuals do not hold a full reference to the [class@Foundry.Context].
Instead it is a weak reference which may be upgraded to a full reference when calling [method@Foundry.Contextual.dup_context].

## Inhibitor

Use [class@Foundry.Inhibitor] to prevent the shutdown of a context during a long running operation.
Calls to [method@Foundry.Context.shutdown] will be asynchornously delayed until the inhibitor as released the shutdown lock.


# Services

Many features in Foundry are implemented through independent services.
The [class@Foundry.Service] base-class is inhereted by these services and attached to the [class@Foundry.Context].

You may implement your own services in your application using libfoundry.
Register the dynamically by calling [method@Foundry.Context.dup_service_typed].


# SDK Management

 * [class@Foundry.SdkManager]
 * [class@Foundry.SdkProvider]
 * [class@Foundry.Sdk]


# Building

 * [class@Foundry.BuildManager]
 * [class@Foundry.BuildPipeline]
 * [class@Foundry.BuildAddin]
 * [class@Foundry.CommandStage]
 * [class@Foundry.BuildProgress]
 * [class@Foundry.BuildFlags]
 * [class@Foundry.CompileCommands]
 * [class@Foundry.DeployStrategy]
 * [class@Foundry.LinkedPipelineStage]


# Running

 * [class@Foundry.RunManager]
 * [class@Foundry.RunTool]


# Build Configuration

 * [class@Foundry.ConfigManager]
 * [class@Foundry.ConfigProvider]
 * [class@Foundry.Config]


# Commands and Subprocesses

 * [class@Foundry.ProcessLauncher]
 * [class@Foundry.UnixFDMap]
 * [class@Foundry.TerminalLauncher]
 * [class@Foundry.CommandManager]
 * [class@Foundry.CommandProvider]
 * [class@Foundry.Command]


# CLI Commands

 * [class@Foundry.CommandLine]
 * [class@Foundry.CliCommandTree]
 * [struct@Foundry.CliCommand]


# Devices

 * [class@Foundry.DeviceManager]
 * [class@Foundry.DeviceProvider]
 * [class@Foundry.DeviceInfo]
 * [class@Foundry.Device]
 * [class@Foundry.LocalDevice]


# Diagnostics

 * [class@Foundry.DiagnosticManager]
 * [class@Foundry.DiagnosticProvider]
 * [class@Foundry.Diagnostic]


# Symbol Extraction

 * [class@Foundry.Symbol]
 * [class@Foundry.SymbolProvider]


# File Management

 * [class@Foundry.FileManager]
 * [class@Foundry.DirectoryListing]
 * [class@Foundry.DirectoryItem]
 * [class@Foundry.DirectoryReaper]
 * [class@Foundry.FileMonitor]
 * [class@Foundry.FileMonitorEvent]


# Settings

 * [class@Foundry.Settings]

Foundry supports robust hierarchical settings which puts the user in control.
The settings are applied in the following order from highest-to-lowest priority.
The first "layer" where a setting has been applied will take priority.

 * User setting with a specific projects `.foundry/user/settings.keyfile`
 * Project setting shared with collaborators in `.foundry/project/settings.keyfile`
 * Application-wide user preferences
 * Application defaults

You may use the `foundry settings` command-line tool or [class@Foundry.Settings] to modify settings at any of these layers.


# Documentation

 * [class@Foundry.DocumentationBundle]
 * [class@Foundry.DocumentationManager]
 * [class@Foundry.DocumentationMatches]
 * [class@Foundry.DocumentationProvider]
 * [class@Foundry.DocumentationQuery]
 * [class@Foundry.DocumentationRoot]
 * [class@Foundry.Documentation]


# Text Editing

 * [class@Foundry.TextManager]
 * [class@Foundry.TextDocument]
 * [class@Foundry.TextDocumentAddin]
 * [class@Foundry.TextBufferProvider]
 * [iface@Foundry.TextBuffer]
 * [class@Foundry.TextEdit]
 * [class@Foundry.SimpleTextBuffer]
 * [class@Foundry.CompletionProvider]
 * [class@Foundry.CompletionProposal]
 * [class@Foundry.CompletionRequest]
 * [class@Foundry.HoverProvider]
 * [class@Foundry.CodeAction]
 * [class@Foundry.OnTypeFormatter]
 * [class@Foundry.OnTypeDiagnostics]
 * [class@Foundry.TextFormatter]
 * [class@Foundry.TextSettings]
 * [class@Foundry.TextSettingsProvider]


# Language Server Protocol

 * [class@Foundry.LspManager]
 * [class@Foundry.LspProvider]
 * [class@Foundry.LspServer]


# Search

 * [class@Foundry.SearchManager]
 * [class@Foundry.SearchProvider]
 * [class@Foundry.SearchRequest]
 * [class@Foundry.SearchResult]


# Flatpak

 * [class@Foundry.FlatpakManifest]


# Version Control

 * [class@Foundry.VcsManager]
 * [class@Foundry.VcsProvider]
 * [class@Foundry.Vcs]
 * [class@Foundry.NoVcs]
 * [class@Foundry.GitVcs]
 * [class@Foundry.VcsBlame]
 * [class@Foundry.VcsTag]
 * [class@Foundry.VcsBranch]
 * [class@Foundry.VcsReference]
 * [class@Foundry.VcsRemote]
 * [class@Foundry.VcsCommit]
 * [class@Foundry.VcsTree]
 * [class@Foundry.VcsDiff]
 * [class@Foundry.VcsDelta]
 * [class@Foundry.VcsSignature]
 * [class@Foundry.VcsLineChanges]


# Operations & Logging

 * [class@Foundry.OperationManager]
 * [class@Foundry.Operation]
 * [class@Foundry.LogManager]
 * [class@Foundry.AuthProvider]


# Unit Testing

 * [class@Foundry.TestManager]
 * [class@Foundry.TestProvider]
 * [class@Foundry.Test]


# Dependency Tracking

 * [class@Foundry.DependencyManager]
 * [class@Foundry.DependencyProvider]
 * [class@Foundry.Dependency]


# Debuggers

 * [class@Foundry.DebuggerManager]
 * [class@Foundry.DebuggerProvider]
 * [class@Foundry.Debugger]
 * [class@Foundry.DebuggerModule]
 * [class@Foundry.DebuggerTarget]
 * [class@Foundry.DebuggerTargetCommand]
 * [class@Foundry.DebuggerTargetProcess]
 * [class@Foundry.DebuggerTargetRemote]
 * [class@Foundry.DebuggerThread]
 * [class@Foundry.DebuggerThreadGroup]
 * [class@Foundry.DebuggerStackFrame]
 * [class@Foundry.DebuggerTrap]
 * [class@Foundry.DebuggerBreakpoint]
 * [class@Foundry.DebuggerWatchpoint]
 * [class@Foundry.DebuggerCountpoint]
 * [class@Foundry.DebuggerSource]
 * [class@Foundry.DebuggerMappedRegion]
 * [class@Foundry.DebuggerInstruction]
 * [class@Foundry.DebuggerEvent]
 * [class@Foundry.DebuggerStopEvent]
 * [class@Foundry.DebuggerVariable]


# Large Language Models

 * [class@Foundry.LlmManager]
 * [class@Foundry.LlmProvider]
 * [class@Foundry.LlmModel]
 * [class@Foundry.LlmCompletion]
 * [class@Foundry.LlmCompletionChunk]
 * [class@Foundry.LlmConversation]
 * [class@Foundry.LlmMessage]
 * [class@Foundry.LlmTool]
 * [class@Foundry.LlmToolCall]

