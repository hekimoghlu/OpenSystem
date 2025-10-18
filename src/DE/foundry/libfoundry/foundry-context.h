/* foundry-context.h
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

#include <glib-object.h>

#include <libdex.h>

#include "libfoundry-config.h"

#include "foundry-types.h"
#include "foundry-version-macros.h"

G_BEGIN_DECLS

#define FOUNDRY_CONTEXT_ERROR      (foundry_context_error_quark())
#define FOUNDRY_TYPE_CONTEXT       (foundry_context_get_type())
#define FOUNDRY_TYPE_CONTEXT_FLAGS (foundry_context_flags_get_type())

typedef enum _FoundryContextFlags
{
  FOUNDRY_CONTEXT_FLAGS_NONE   = 0,
  FOUNDRY_CONTEXT_FLAGS_CREATE = 1 << 0,
  FOUNDRY_CONTEXT_FLAGS_SHARED = 1 << 1,
} FoundryContextFlags;

typedef enum _FoundryContextError
{
  FOUNDRY_CONTEXT_ERROR_IN_SHUTDOWN = 1,
} FoundryContextError;

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_FINAL_TYPE (FoundryContext, foundry_context, FOUNDRY, CONTEXT, GObject)

FOUNDRY_AVAILABLE_IN_ALL
GQuark                       foundry_context_error_quark               (void) G_GNUC_CONST;
FOUNDRY_AVAILABLE_IN_ALL
GType                        foundry_context_flags_get_type            (void) G_GNUC_CONST;
FOUNDRY_AVAILABLE_IN_ALL
DexFuture                   *foundry_context_discover                  (const char          *path,
                                                                        DexCancellable      *cancellable) G_GNUC_WARN_UNUSED_RESULT;
FOUNDRY_AVAILABLE_IN_ALL
DexFuture                   *foundry_context_new                       (const char          *foundry_dir,
                                                                        const char          *project_dir,
                                                                        FoundryContextFlags  flags,
                                                                        DexCancellable      *cancellable) G_GNUC_WARN_UNUSED_RESULT;
FOUNDRY_AVAILABLE_IN_ALL
DexFuture                   *foundry_context_new_for_user              (DexCancellable      *cancellable) G_GNUC_WARN_UNUSED_RESULT;
FOUNDRY_AVAILABLE_IN_ALL
DexFuture                   *foundry_context_save                      (FoundryContext      *self,
                                                                        DexCancellable      *cancellable) G_GNUC_WARN_UNUSED_RESULT;
FOUNDRY_AVAILABLE_IN_ALL
GFile                       *foundry_context_dup_state_directory       (FoundryContext      *self);
FOUNDRY_AVAILABLE_IN_ALL
GFile                       *foundry_context_dup_project_directory     (FoundryContext      *self);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture                   *foundry_context_shutdown                  (FoundryContext      *self) G_GNUC_WARN_UNUSED_RESULT;
FOUNDRY_AVAILABLE_IN_ALL
gboolean                     foundry_context_network_allowed           (FoundryContext      *self);
FOUNDRY_AVAILABLE_IN_ALL
void                         foundry_context_log                       (FoundryContext      *self,
                                                                        const char          *domain,
                                                                        GLogLevelFlags       severity,
                                                                        const char          *format,
                                                                        ...) G_GNUC_PRINTF(4, 5);
FOUNDRY_AVAILABLE_IN_ALL
void                         foundry_context_logv                      (FoundryContext      *self,
                                                                        const char          *domain,
                                                                        GLogLevelFlags       severity,
                                                                        const char          *format,
                                                                        va_list              args);
FOUNDRY_AVAILABLE_IN_ALL
gboolean                     foundry_context_is_shared                 (FoundryContext      *self);
FOUNDRY_AVAILABLE_IN_ALL
char                        *foundry_context_dup_build_system          (FoundryContext      *self);
FOUNDRY_AVAILABLE_IN_ALL
FoundryBuildManager         *foundry_context_dup_build_manager         (FoundryContext      *self);
FOUNDRY_AVAILABLE_IN_ALL
FoundryCommandManager       *foundry_context_dup_command_manager       (FoundryContext      *self);
FOUNDRY_AVAILABLE_IN_ALL
FoundryConfigManager        *foundry_context_dup_config_manager        (FoundryContext      *self);
FOUNDRY_AVAILABLE_IN_ALL
FoundryDependencyManager    *foundry_context_dup_dependency_manager    (FoundryContext      *self);
FOUNDRY_AVAILABLE_IN_ALL
FoundryDeviceManager        *foundry_context_dup_device_manager        (FoundryContext      *self);
FOUNDRY_AVAILABLE_IN_ALL
FoundryDiagnosticManager    *foundry_context_dup_diagnostic_manager    (FoundryContext      *self);
FOUNDRY_AVAILABLE_IN_ALL
FoundryFileManager          *foundry_context_dup_file_manager          (FoundryContext      *self);
FOUNDRY_AVAILABLE_IN_ALL
FoundryRunManager           *foundry_context_dup_run_manager           (FoundryContext      *self);
FOUNDRY_AVAILABLE_IN_ALL
FoundrySdkManager           *foundry_context_dup_sdk_manager           (FoundryContext      *self);
FOUNDRY_AVAILABLE_IN_ALL
FoundrySearchManager        *foundry_context_dup_search_manager        (FoundryContext      *self);
FOUNDRY_AVAILABLE_IN_ALL
FoundryTestManager          *foundry_context_dup_test_manager          (FoundryContext      *self);
FOUNDRY_AVAILABLE_IN_ALL
FoundryDBusService          *foundry_context_dup_dbus_service          (FoundryContext      *self);
FOUNDRY_AVAILABLE_IN_ALL
FoundryLogManager           *foundry_context_dup_log_manager           (FoundryContext      *self);
FOUNDRY_AVAILABLE_IN_ALL
FoundryOperationManager     *foundry_context_dup_operation_manager     (FoundryContext      *self);
FOUNDRY_AVAILABLE_IN_ALL
FoundryTweakManager         *foundry_context_dup_tweak_manager         (FoundryContext      *self);
FOUNDRY_AVAILABLE_IN_ALL
gpointer                     foundry_context_dup_service_typed         (FoundryContext      *self,
                                                                        GType                service_type);
FOUNDRY_AVAILABLE_IN_ALL
GListModel                  *foundry_context_list_services             (FoundryContext      *self);
FOUNDRY_AVAILABLE_IN_ALL
GActionGroup                *foundry_context_dup_action_group          (FoundryContext      *self);
FOUNDRY_AVAILABLE_IN_ALL
FoundrySettings             *foundry_context_load_settings             (FoundryContext      *self,
                                                                        const char          *schema_id,
                                                                        const char          *schema_path);
FOUNDRY_AVAILABLE_IN_ALL
FoundrySettings             *foundry_context_load_project_settings     (FoundryContext      *self);
FOUNDRY_AVAILABLE_IN_ALL
char                        *foundry_context_cache_filename            (FoundryContext      *self,
                                                                        ...) G_GNUC_NULL_TERMINATED;
FOUNDRY_AVAILABLE_IN_ALL
GFile                       *foundry_context_cache_file                (FoundryContext      *self,
                                                                        ...) G_GNUC_NULL_TERMINATED;
FOUNDRY_AVAILABLE_IN_1_1
char                        *foundry_context_tmp_filename              (FoundryContext      *self,
                                                                        ...) G_GNUC_NULL_TERMINATED;
FOUNDRY_AVAILABLE_IN_ALL
char                        *foundry_context_dup_title                 (FoundryContext      *self);
FOUNDRY_AVAILABLE_IN_ALL
void                         foundry_context_set_title                 (FoundryContext      *self,
                                                                        const char          *title);
FOUNDRY_AVAILABLE_IN_ALL
FoundryLicense              *foundry_context_dup_default_license       (FoundryContext      *self);

#ifdef FOUNDRY_FEATURE_DEBUGGER
FOUNDRY_AVAILABLE_IN_ALL
FoundryDebuggerManager      *foundry_context_dup_debugger_manager      (FoundryContext      *self);
#endif

#ifdef FOUNDRY_FEATURE_DOCS
FOUNDRY_AVAILABLE_IN_ALL
FoundryDocumentationManager *foundry_context_dup_documentation_manager (FoundryContext      *self);
#endif

#ifdef FOUNDRY_FEATURE_LLM
FOUNDRY_AVAILABLE_IN_ALL
FoundryLlmManager           *foundry_context_dup_llm_manager           (FoundryContext      *self);
#endif

#ifdef FOUNDRY_FEATURE_LSP
FOUNDRY_AVAILABLE_IN_ALL
FoundryLspManager           *foundry_context_dup_lsp_manager           (FoundryContext      *self);
#endif

#ifdef FOUNDRY_FEATURE_TEXT
FOUNDRY_AVAILABLE_IN_ALL
FoundryTextManager          *foundry_context_dup_text_manager          (FoundryContext      *self);
#endif

#ifdef FOUNDRY_FEATURE_VCS
FOUNDRY_AVAILABLE_IN_ALL
FoundryVcsManager           *foundry_context_dup_vcs_manager           (FoundryContext      *self);
#endif

#define FOUNDRY_DEBUG(context, format, ...) \
  foundry_context_log((context), G_LOG_DOMAIN, G_LOG_LEVEL_DEBUG, format, __VA_ARGS__)
#define FOUNDRY_MESSAGE(context, format, ...) \
  foundry_context_log((context), G_LOG_DOMAIN, G_LOG_LEVEL_MESSAGE, format, __VA_ARGS__)
#define FOUNDRY_WARNING(context, format, ...) \
  foundry_context_log((context), G_LOG_DOMAIN, G_LOG_LEVEL_WARNING, format, __VA_ARGS__)

G_END_DECLS
