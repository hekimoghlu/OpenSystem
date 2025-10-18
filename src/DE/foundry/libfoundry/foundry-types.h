/* foundry-types.h
 *
 * Copyright 2024 Christian Hergert <chergert@redhat.com>
 *
 * This library is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation; either version 2.1 of the
 * License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

#pragma once

#include <glib.h>

#include <libfoundry-config.h>

G_BEGIN_DECLS

/*<private>
 * FOUNDRY_DECLARE_INTERNAL_TYPE:
 * @ModuleObjName: The name of the new type, in camel case (like GtkWidget)
 * @module_obj_name: The name of the new type in lowercase, with words
 *  separated by '_' (like 'gtk_widget')
 * @MODULE: The name of the module, in all caps (like 'GTK')
 * @OBJ_NAME: The bare name of the type, in all caps (like 'WIDGET')
 * @ParentName: the name of the parent type, in camel case (like GtkWidget)
 *
 * A convenience macro for emitting the usual declarations in the
 * header file for a type which is intended to be subclassed only
 * by internal consumers.
 *
 * This macro differs from %G_DECLARE_DERIVABLE_TYPE and %G_DECLARE_FINAL_TYPE
 * by declaring a type that is only derivable internally. Internal users can
 * derive this type, assuming they have access to the instance and class
 * structures; external users will not be able to subclass this type.
 */
#define FOUNDRY_DECLARE_INTERNAL_TYPE(ModuleObjName, module_obj_name, MODULE, OBJ_NAME, ParentName)   \
  GType module_obj_name##_get_type (void);                                                            \
  G_GNUC_BEGIN_IGNORE_DEPRECATIONS                                                                    \
  typedef struct _##ModuleObjName ModuleObjName;                                                      \
  typedef struct _##ModuleObjName##Class ModuleObjName##Class;                                        \
                                                                                                      \
  _GLIB_DEFINE_AUTOPTR_CHAINUP (ModuleObjName, ParentName)                                            \
  G_DEFINE_AUTOPTR_CLEANUP_FUNC (ModuleObjName##Class, g_type_class_unref)                            \
                                                                                                      \
  G_GNUC_UNUSED static inline ModuleObjName * MODULE##_##OBJ_NAME (gpointer ptr) {                    \
    return G_TYPE_CHECK_INSTANCE_CAST (ptr, module_obj_name##_get_type (), ModuleObjName); }          \
  G_GNUC_UNUSED static inline ModuleObjName##Class * MODULE##_##OBJ_NAME##_CLASS (gpointer ptr) {     \
    return G_TYPE_CHECK_CLASS_CAST (ptr, module_obj_name##_get_type (), ModuleObjName##Class); }      \
  G_GNUC_UNUSED static inline gboolean MODULE##_IS_##OBJ_NAME (gpointer ptr) {                        \
    return G_TYPE_CHECK_INSTANCE_TYPE (ptr, module_obj_name##_get_type ()); }                         \
  G_GNUC_UNUSED static inline gboolean MODULE##_IS_##OBJ_NAME##_CLASS (gpointer ptr) {                \
    return G_TYPE_CHECK_CLASS_TYPE (ptr, module_obj_name##_get_type ()); }                            \
  G_GNUC_UNUSED static inline ModuleObjName##Class * MODULE##_##OBJ_NAME##_GET_CLASS (gpointer ptr) { \
    return G_TYPE_INSTANCE_GET_CLASS (ptr, module_obj_name##_get_type (), ModuleObjName##Class); }    \
  G_GNUC_END_IGNORE_DEPRECATIONS

typedef struct _FoundryAuthProvider              FoundryAuthProvider;
typedef struct _FoundryBuildAddin                FoundryBuildAddin;
typedef struct _FoundryBuildFlags                FoundryBuildFlags;
typedef struct _FoundryBuildManager              FoundryBuildManager;
typedef struct _FoundryBuildPipeline             FoundryBuildPipeline;
typedef struct _FoundryBuildProgress             FoundryBuildProgress;
typedef struct _FoundryBuildStage                FoundryBuildStage;
typedef struct _FoundryBuildTarget               FoundryBuildTarget;
typedef struct _FoundryCliCommand                FoundryCliCommand;
typedef struct _FoundryCliTool                   FoundryCliTool;
typedef struct _FoundryCodeAction                FoundryCodeAction;
typedef struct _FoundryCommandLine               FoundryCommandLine;
typedef struct _FoundryCommand                   FoundryCommand;
typedef struct _FoundryCommandManager            FoundryCommandManager;
typedef struct _FoundryCommandProvider           FoundryCommandProvider;
typedef struct _FoundryCommandStage              FoundryCommandStage;
typedef struct _FoundryCompileCommands           FoundryCompileCommands;
typedef struct _FoundryConfig                    FoundryConfig;
typedef struct _FoundryConfigManager             FoundryConfigManager;
typedef struct _FoundryConfigProvider            FoundryConfigProvider;
typedef struct _FoundryContext                   FoundryContext;
typedef struct _FoundryContextual                FoundryContextual;
typedef struct _FoundryDBusService               FoundryDBusService;
typedef struct _FoundryDependency                FoundryDependency;
typedef struct _FoundryDependencyManager         FoundryDependencyManager;
typedef struct _FoundryDependencyProvider        FoundryDependencyProvider;
typedef struct _FoundryDeployStrategy            FoundryDeployStrategy;
typedef struct _FoundryDevice                    FoundryDevice;
typedef enum   _FoundryDeviceChassis             FoundryDeviceChassis;
typedef struct _FoundryDeviceInfo                FoundryDeviceInfo;
typedef struct _FoundryDeviceProvider            FoundryDeviceProvider;
typedef struct _FoundryDeviceManager             FoundryDeviceManager;
typedef struct _FoundryDiagnostic                FoundryDiagnostic;
typedef struct _FoundryDiagnosticBuilder         FoundryDiagnosticBuilder;
typedef struct _FoundryDiagnosticFix             FoundryDiagnosticFix;
typedef struct _FoundryDiagnosticProvider        FoundryDiagnosticProvider;
typedef struct _FoundryDiagnosticTool            FoundryDiagnosticTool;
typedef struct _FoundryDiagnosticManager         FoundryDiagnosticManager;
typedef struct _FoundryDirectoryListing          FoundryDirectoryListing;
typedef struct _FoundryDirectoryItem             FoundryDirectoryItem;
typedef struct _FoundryDirectoryReaper           FoundryDirectoryReaper;
typedef struct _FoundryExtension                 FoundryExtension;
typedef struct _FoundryExtensionSet              FoundryExtensionSet;
typedef struct _FoundryFileManager               FoundryFileManager;
typedef struct _FoundryInhibitor                 FoundryInhibitor;
typedef struct _FoundryInput                     FoundryInput;
typedef struct _FoundryInputChoice               FoundryInputChoice;
typedef struct _FoundryInputCombo                FoundryInputCombo;
typedef struct _FoundryInputFile                 FoundryInputFile;
typedef struct _FoundryInputFont                 FoundryInputFont;
typedef struct _FoundryInputGroup                FoundryInputGroup;
typedef struct _FoundryInputPassword             FoundryInputPassword;
typedef struct _FoundryInputSpin                 FoundryInputSpin;
typedef struct _FoundryInputSwitch               FoundryInputSwitch;
typedef struct _FoundryInputText                 FoundryInputText;
typedef struct _FoundryInputValidator            FoundryInputValidator;
typedef struct _FoundryInputValidatorDelegate    FoundryInputValidatorDelegate;
typedef struct _FoundryInputValidatorRegex       FoundryInputValidatorRegex;
typedef struct _FoundryLanguage                  FoundryLanguage;
typedef struct _FoundryLanguageGuesser           FoundryLanguageGuesser;
typedef struct _FoundryLicense                   FoundryLicense;
typedef struct _FoundryLogManager                FoundryLogManager;
typedef struct _FoundryLogMessage                FoundryLogMessage;
typedef struct _FoundryMarkup                    FoundryMarkup;
typedef struct _FoundryMcpClient                 FoundryMcpClient;
typedef struct _FoundryOperation                 FoundryOperation;
typedef struct _FoundryOperationManager          FoundryOperationManager;
typedef struct _FoundryPathCache                 FoundryPathCache;
typedef struct _FoundryPipeline                  FoundryPipeline;
typedef struct _FoundryPluginManager             FoundryPluginManager;
typedef struct _FoundryProcessLauncher           FoundryProcessLauncher;
typedef struct _FoundryPtyDiagnostics            FoundryPtyDiagnostics;
typedef struct _FoundryRunManager                FoundryRunManager;
typedef struct _FoundryRunTool                   FoundryRunTool;
typedef struct _FoundrySdk                       FoundrySdk;
typedef struct _FoundrySdkManager                FoundrySdkManager;
typedef struct _FoundrySdkProvider               FoundrySdkProvider;
typedef struct _FoundrySettings                  FoundrySettings;
typedef struct _FoundrySymbol                    FoundrySymbol;
typedef struct _FoundrySymbolProvider            FoundrySymbolProvider;
typedef struct _FoundrySearchManager             FoundrySearchManager;
typedef struct _FoundrySearchProvider            FoundrySearchProvider;
typedef struct _FoundrySearchRequest             FoundrySearchRequest;
typedef struct _FoundrySearchResult              FoundrySearchResult;
typedef struct _FoundryService                   FoundryService;
typedef struct _FoundryTest                      FoundryTest;
typedef struct _FoundryTestManager               FoundryTestManager;
typedef struct _FoundryTestProvider              FoundryTestProvider;
typedef struct _FoundryTextEdit                  FoundryTextEdit;
typedef struct _FoundryTriplet                   FoundryTriplet;
typedef struct _FoundryTtyAuthProvider           FoundryTtyAuthProvider;
typedef struct _FoundryTweak                     FoundryTweak;
typedef struct _FoundryTweakInfo                 FoundryTweakInfo;
typedef struct _FoundryTweakPath                 FoundryTweakPath;
typedef struct _FoundryTweakManager              FoundryTweakManager;
typedef struct _FoundryTweakProvider             FoundryTweakProvider;
typedef struct _FoundryUnixFDMap                 FoundryUnixFDMap;

#ifdef FOUNDRY_FEATURE_DAP
typedef struct _FoundryDapDebugger               FoundryDapDebugger;
#endif

#ifdef FOUNDRY_FEATURE_DEBUGGER
typedef struct _FoundryDebugger                  FoundryDebugger;
typedef struct _FoundryDebuggerActions           FoundryDebuggerActions;
typedef struct _FoundryDebuggerLogMessage        FoundryDebuggerLogMessage;
typedef struct _FoundryDebuggerBreakpoint        FoundryDebuggerBreakpoint;
typedef struct _FoundryDebuggerCountpoint        FoundryDebuggerCountpoint;
typedef struct _FoundryDebuggerInstruction       FoundryDebuggerInstruction;
typedef struct _FoundryDebuggerEvent             FoundryDebuggerEvent;
typedef struct _FoundryDebuggerManager           FoundryDebuggerManager;
typedef struct _FoundryDebuggerMappedRegion      FoundryDebuggerMappedRegion;
typedef struct _FoundryDebuggerModule            FoundryDebuggerModule;
typedef struct _FoundryDebuggerProvider          FoundryDebuggerProvider;
typedef struct _FoundryDebuggerSource            FoundryDebuggerSource;
typedef struct _FoundryDebuggerStopEvent         FoundryDebuggerStopEvent;
typedef struct _FoundryDebuggerTarget            FoundryDebuggerTarget;
typedef struct _FoundryDebuggerTargetCommand     FoundryDebuggerTargetCommand;
typedef struct _FoundryDebuggerTargetProcess     FoundryDebuggerTargetProcess;
typedef struct _FoundryDebuggerTargetRemote      FoundryDebuggerTargetRemote;
typedef struct _FoundryDebuggerThread            FoundryDebuggerThread;
typedef struct _FoundryDebuggerThreadGroup       FoundryDebuggerThreadGroup;
typedef struct _FoundryDebuggerTrap              FoundryDebuggerTrap;
typedef struct _FoundryDebuggerTrapParams        FoundryDebuggerTrapParams;
typedef struct _FoundryDebuggerVariable          FoundryDebuggerVariable;
typedef struct _FoundryDebuggerWatchpoint        FoundryDebuggerWatchpoint;

typedef enum _FoundryDebuggerMovement
{
  FOUNDRY_DEBUGGER_MOVEMENT_START,
  FOUNDRY_DEBUGGER_MOVEMENT_CONTINUE,
  FOUNDRY_DEBUGGER_MOVEMENT_STEP_IN,
  FOUNDRY_DEBUGGER_MOVEMENT_STEP_OVER,
  FOUNDRY_DEBUGGER_MOVEMENT_STEP_OUT,
} FoundryDebuggerMovement;

typedef enum _FoundryDebuggerTrapDisposition
{
  FOUNDRY_DEBUGGER_TRAP_KEEP = 0,
  FOUNDRY_DEBUGGER_TRAP_DISABLE,
  FOUNDRY_DEBUGGER_TRAP_REMOVE_NEXT_HIT,
  FOUNDRY_DEBUGGER_TRAP_REMOVE_NEXT_STOP,
} FoundryDebuggerTrapDisposition;

typedef enum _FoundryDebuggerWatchAccess
{
  FOUNDRY_DEBUGGER_WATCH_NONE      = 0,
  FOUNDRY_DEBUGGER_WATCH_READ      = 1 << 0,
  FOUNDRY_DEBUGGER_WATCH_WRITE     = 1 << 1,
  FOUNDRY_DEBUGGER_WATCH_READWRITE = (FOUNDRY_DEBUGGER_WATCH_READ | FOUNDRY_DEBUGGER_WATCH_WRITE),
} FoundryDebuggerWatchAccess;

typedef enum _FoundryDebuggerTrapKind
{
  FOUNDRY_DEBUGGER_TRAP_KIND_BREAKPOINT,
  FOUNDRY_DEBUGGER_TRAP_KIND_WATCHPOINT,
  FOUNDRY_DEBUGGER_TRAP_KIND_COUNTPOINT,
} FoundryDebuggerTrapKind;

typedef enum _FoundryDebuggerStopReason
{
  FOUNDRY_DEBUGGER_STOP_UNKNOWN,
  FOUNDRY_DEBUGGER_STOP_BREAKPOINT_HIT,
  FOUNDRY_DEBUGGER_STOP_EXITED,
  FOUNDRY_DEBUGGER_STOP_EXITED_NORMALLY,
  FOUNDRY_DEBUGGER_STOP_EXITED_SIGNALED,
  FOUNDRY_DEBUGGER_STOP_FUNCTION_FINISHED,
  FOUNDRY_DEBUGGER_STOP_LOCATION_REACHED,
  FOUNDRY_DEBUGGER_STOP_SIGNAL_RECEIVED,
  FOUNDRY_DEBUGGER_STOP_CATCH,
} FoundryDebuggerStopReason;
#endif

#ifdef FOUNDRY_FEATURE_DOCS
typedef struct _FoundryDocumentation             FoundryDocumentation;
typedef struct _FoundryDocumentationBundle       FoundryDocumentationBundle;
typedef struct _FoundryDocumentationManager      FoundryDocumentationManager;
typedef struct _FoundryDocumentationMatches      FoundryDocumentationMatches;
typedef struct _FoundryDocumentationProvider     FoundryDocumentationProvider;
typedef struct _FoundryDocumentationQuery        FoundryDocumentationQuery;
#endif

#ifdef FOUNDRY_FEATURE_LLM
typedef struct _FoundryLlmConversation           FoundryLlmConversation;
typedef struct _FoundryLlmMessage                FoundryLlmMessage;
typedef struct _FoundryLlmCompletion             FoundryLlmCompletion;
typedef struct _FoundryLlmCompletionChunk        FoundryLlmCompletionChunk;
typedef struct _FoundryLlmManager                FoundryLlmManager;
typedef struct _FoundryLlmMessage                FoundryLlmMessage;
typedef struct _FoundryLlmModel                  FoundryLlmModel;
typedef struct _FoundryLlmProvider               FoundryLlmProvider;
typedef struct _FoundryLlmTool                   FoundryLlmTool;
typedef struct _FoundryLlmToolCall               FoundryLlmToolCall;
typedef struct _FoundrySimpleLlmMessage          FoundrySimpleLlmMessage;
#endif

#ifdef FOUNDRY_FEATURE_LSP
typedef struct _FoundryLspClient                 FoundryLspClient;
typedef struct _FoundryLspCompletionProvider     FoundryLspCompletionProvider;
typedef struct _FoundryLspManager                FoundryLspManager;
typedef struct _FoundryLspProvider               FoundryLspProvider;
typedef struct _FoundryLspServer                 FoundryLspServer;
#endif

#ifdef FOUNDRY_FEATURE_TEXT
typedef struct _FoundryCompletionProposal        FoundryCompletionProposal;
typedef struct _FoundryCompletionProvider        FoundryCompletionProvider;
typedef struct _FoundryCompletionRequest         FoundryCompletionRequest;
typedef struct _FoundryHoverProvider             FoundryHoverProvider;
typedef struct _FoundryOnTypeDiagnostics         FoundryOnTypeDiagnostics;
typedef struct _FoundryOnTypeFormatter           FoundryOnTypeFormatter;
typedef struct _FoundryRenameProvider            FoundryRenameProvider;
typedef struct _FoundryTextBuffer                FoundryTextBuffer;
typedef struct _FoundryTextBufferProvider        FoundryTextBufferProvider;
typedef struct _FoundryTextDocument              FoundryTextDocument;
typedef struct _FoundryTextDocumentAddin         FoundryTextDocumentAddin;
typedef struct _FoundryTextFormatter             FoundryTextFormatter;
typedef struct _FoundryTextIter                  FoundryTextIter;
typedef struct _FoundryTextManager               FoundryTextManager;
typedef struct _FoundryTextSettings              FoundryTextSettings;
typedef struct _FoundryTextSettingsProvider      FoundryTextSettingsProvider;
#endif

#ifdef FOUNDRY_FEATURE_TEMPLATES
typedef struct _FoundryCodeTemplate              FoundryCodeTemplate;
typedef struct _FoundryProjectTemplate           FoundryProjectTemplate;
typedef struct _FoundryTemplate                  FoundryTemplate;
typedef struct _FoundryTemplateManager           FoundryTemplateManager;
typedef struct _FoundryTemplateOutput           FoundryTemplateOutput;
typedef struct _FoundryTemplateProvider          FoundryTemplateProvider;
#endif

#ifdef FOUNDRY_FEATURE_TERMINAL
typedef struct _FoundryTerminalLauncher          FoundryTerminalLauncher;
#endif

#ifdef FOUNDRY_FEATURE_VCS
typedef struct _FoundryVcs                       FoundryVcs;
typedef struct _FoundryVcsBlame                  FoundryVcsBlame;
typedef struct _FoundryVcsBranch                 FoundryVcsBranch;
typedef struct _FoundryVcsCommit                 FoundryVcsCommit;
typedef struct _FoundryVcsDelta                  FoundryVcsDelta;
typedef struct _FoundryVcsDiff                   FoundryVcsDiff;
typedef struct _FoundryVcsFile                   FoundryVcsFile;
typedef struct _FoundryVcsLineChanges            FoundryVcsLineChanges;
typedef struct _FoundryVcsManager                FoundryVcsManager;
typedef struct _FoundryVcsProvider               FoundryVcsProvider;
typedef struct _FoundryVcsReference              FoundryVcsReference;
typedef struct _FoundryVcsRemote                 FoundryVcsRemote;
typedef struct _FoundryVcsSignature              FoundryVcsSignature;
typedef struct _FoundryVcsStats                  FoundryVcsStats;
typedef struct _FoundryVcsTag                    FoundryVcsTag;
typedef struct _FoundryVcsTree                   FoundryVcsTree;

typedef enum _FoundryVcsFileStatus
{
  FOUNDRY_VCS_FILE_STATUS_CURRENT           = 0,
  FOUNDRY_VCS_FILE_STATUS_MODIFIED_IN_STAGE = 1 << 1,
  FOUNDRY_VCS_FILE_STATUS_MODIFIED_IN_TREE  = 1 << 2,
  FOUNDRY_VCS_FILE_STATUS_NEW_IN_STAGE      = 1 << 3,
  FOUNDRY_VCS_FILE_STATUS_NEW_IN_TREE       = 1 << 4,
  FOUNDRY_VCS_FILE_STATUS_DELETED_IN_STAGE  = 1 << 5,
  FOUNDRY_VCS_FILE_STATUS_DELETED_IN_TREE   = 1 << 6,
} FoundryVcsFileStatus;
#endif

typedef enum _FoundryBuildPipelinePhase
{
  FOUNDRY_BUILD_PIPELINE_PHASE_NONE         = 0,
  FOUNDRY_BUILD_PIPELINE_PHASE_PREPARE      = 1 << 0,
  FOUNDRY_BUILD_PIPELINE_PHASE_DOWNLOADS    = 1 << 1,
  FOUNDRY_BUILD_PIPELINE_PHASE_DEPENDENCIES = 1 << 2,
  FOUNDRY_BUILD_PIPELINE_PHASE_AUTOGEN      = 1 << 3,
  FOUNDRY_BUILD_PIPELINE_PHASE_CONFIGURE    = 1 << 4,
  FOUNDRY_BUILD_PIPELINE_PHASE_BUILD        = 1 << 6,
  FOUNDRY_BUILD_PIPELINE_PHASE_INSTALL      = 1 << 7,
  FOUNDRY_BUILD_PIPELINE_PHASE_COMMIT       = 1 << 8,
  FOUNDRY_BUILD_PIPELINE_PHASE_EXPORT       = 1 << 9,
  FOUNDRY_BUILD_PIPELINE_PHASE_FINAL        = 1 << 10,

  FOUNDRY_BUILD_PIPELINE_PHASE_BEFORE       = 1 << 28,
  FOUNDRY_BUILD_PIPELINE_PHASE_AFTER        = 1 << 29,
  FOUNDRY_BUILD_PIPELINE_PHASE_FINISHED     = 1 << 30,
  FOUNDRY_BUILD_PIPELINE_PHASE_FAILED       = 1 << 31,
} FoundryBuildPipelinePhase;

typedef enum _FoundryCompletionActivation
{
  FOUNDRY_COMPLETION_ACTIVATION_NONE = 0,
  FOUNDRY_COMPLETION_ACTIVATION_INTERACTIVE,
  FOUNDRY_COMPLETION_ACTIVATION_USER_REQUESTED,
} FoundryCompletionActivation;

G_END_DECLS
