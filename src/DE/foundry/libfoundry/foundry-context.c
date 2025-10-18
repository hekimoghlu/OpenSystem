/* foundry-context.c
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

#include "config.h"

#include <glib/gi18n-lib.h>

#include <libpeas.h>

#define G_SETTINGS_ENABLE_BACKEND
#include <gio/gsettingsbackend.h>

#include "foundry-action-muxer.h"
#include "foundry-build-manager.h"
#include "foundry-command-manager.h"
#include "foundry-config.h"
#include "foundry-config-manager.h"
#include "foundry-context-private.h"
#include "foundry-dbus-service.h"
#include "foundry-debug.h"
#include "foundry-dependency-manager.h"
#include "foundry-device-manager.h"
#include "foundry-diagnostic-manager.h"
#include "foundry-file-manager.h"
#include "foundry-init-private.h"
#include "foundry-license.h"
#include "foundry-log-manager-private.h"
#include "foundry-operation-manager.h"
#include "foundry-run-manager.h"
#include "foundry-sdk-manager.h"
#include "foundry-search-manager.h"
#include "foundry-service-private.h"
#include "foundry-settings.h"
#include "foundry-test-manager.h"
#include "foundry-tweak-manager.h"
#include "foundry-util-private.h"

#ifdef FOUNDRY_FEATURE_DOCS
# include "foundry-documentation-manager.h"
#endif

#ifdef FOUNDRY_FEATURE_DEBUGGER
# include "foundry-debugger-manager.h"
#endif

#ifdef FOUNDRY_FEATURE_LLM
# include "foundry-llm-manager.h"
#endif

#ifdef FOUNDRY_FEATURE_LSP
# include "foundry-lsp-manager.h"
#endif

#ifdef FOUNDRY_FEATURE_TEXT
# include "foundry-text-manager.h"
#endif

#ifdef FOUNDRY_FEATURE_VCS
# include "foundry-vcs-manager.h"
#endif

struct _FoundryContext
{
  GObject            parent_instance;
  GList              link;
  GFile             *project_directory;
  GFile             *state_directory;
  GPtrArray         *services;
  PeasExtensionSet  *service_addins;
  DexFuture         *shutdown;
  DexPromise        *inhibit;
  FoundryLogManager *log_manager;
  FoundrySettings   *project_settings;
  GSettingsBackend  *project_settings_backend;
  GSettingsBackend  *user_settings_backend;
  GHashTable        *settings;
  char              *title;
  guint              inhibit_count;
  guint              is_shared : 1;
};

enum {
  PROP_0,
  PROP_BUILD_MANAGER,
  PROP_BUILD_SYSTEM,
  PROP_COMMAND_MANAGER,
  PROP_CONFIG_MANAGER,
#ifdef FOUNDRY_FEATURE_DEBUGGER
  PROP_DEBUGGER_MANAGER,
#endif
  PROP_DEPENDENCY_MANAGER,
  PROP_DEVICE_MANAGER,
  PROP_DIAGNOSTIC_MANAGER,
#ifdef FOUNDRY_FEATURE_DOCS
  PROP_DOCUMENTATION_MANAGER,
#endif
  PROP_FILE_MANAGER,
#ifdef FOUNDRY_FEATURE_LLM
  PROP_LLM_MANAGER,
#endif
  PROP_LOG_MANAGER,
#ifdef FOUNDRY_FEATURE_LSP
  PROP_LSP_MANAGER,
#endif
  PROP_OPERATION_MANAGER,
  PROP_PROJECT_DIRECTORY,
  PROP_RUN_MANAGER,
  PROP_SDK_MANAGER,
  PROP_SEARCH_MANAGER,
  PROP_STATE_DIRECTORY,
  PROP_TEST_MANAGER,
#ifdef FOUNDRY_FEATURE_TEXT
  PROP_TEXT_MANAGER,
#endif
  PROP_TITLE,
  PROP_TWEAK_MANAGER,
#ifdef FOUNDRY_FEATURE_VCS
  PROP_VCS_MANAGER,
#endif
  N_PROPS
};

G_DEFINE_FINAL_TYPE (FoundryContext, foundry_context, G_TYPE_OBJECT)
G_DEFINE_QUARK (foundry_context_error, foundry_context_error)
G_DEFINE_FLAGS_TYPE (FoundryContextFlags, foundry_context_flags,
                     G_DEFINE_ENUM_VALUE (FOUNDRY_CONTEXT_FLAGS_NONE, "none"),
                     G_DEFINE_ENUM_VALUE (FOUNDRY_CONTEXT_FLAGS_CREATE, "create"))

static GParamSpec *properties[N_PROPS];
static GQueue all_contexts;
G_LOCK_DEFINE_STATIC (all_contexts);

static inline gboolean
foundry_context_in_shutdown (FoundryContext *self)
{
  return self->shutdown != NULL;
}

static DexFuture *
foundry_context_log_failure (DexFuture *future,
                             gpointer   user_data)
{
  g_autoptr(GError) error = NULL;
  g_autoptr(GObject) object = user_data;

  if (NULL == dex_future_get_value (future, &error))
    g_warning ("The object \"%s\" at %p had an error: %s",
               G_OBJECT_TYPE_NAME (object),
               object,
               error->message);

  return dex_ref (future);
}

static void
foundry_context_dispose (GObject *object)
{
  FoundryContext *self = (FoundryContext *)object;

  g_assert (self->inhibit_count == 0);

  g_clear_object (&self->project_settings);

  if (self->services->len > 0)
    g_ptr_array_remove_range (self->services, 0, self->services->len);

  g_hash_table_remove_all (self->settings);

  dex_clear (&self->shutdown);
  dex_clear (&self->inhibit);

  G_OBJECT_CLASS (foundry_context_parent_class)->dispose (object);
}

static void
foundry_context_finalize (GObject *object)
{
  FoundryContext *self = (FoundryContext *)object;

  g_clear_object (&self->project_settings_backend);
  g_clear_object (&self->user_settings_backend);
  g_clear_object (&self->project_directory);
  g_clear_object (&self->state_directory);
  g_clear_object (&self->log_manager);

  g_clear_pointer (&self->services, g_ptr_array_unref);
  g_clear_pointer (&self->settings, g_hash_table_unref);

  g_clear_pointer (&self->title, g_free);

  G_LOCK (all_contexts);
  g_queue_unlink (&all_contexts, &self->link);
  G_UNLOCK (all_contexts);

  G_OBJECT_CLASS (foundry_context_parent_class)->finalize (object);
}

static void
foundry_context_get_property (GObject    *object,
                              guint       prop_id,
                              GValue     *value,
                              GParamSpec *pspec)
{
  FoundryContext *self = FOUNDRY_CONTEXT (object);

  switch (prop_id)
    {
    case PROP_BUILD_MANAGER:
      g_value_take_object (value, foundry_context_dup_build_manager (self));
      break;

    case PROP_BUILD_SYSTEM:
      g_value_take_string (value, foundry_context_dup_build_system (self));
      break;

    case PROP_COMMAND_MANAGER:
      g_value_take_object (value, foundry_context_dup_command_manager (self));
      break;

    case PROP_CONFIG_MANAGER:
      g_value_take_object (value, foundry_context_dup_config_manager (self));
      break;

#ifdef FOUNDRY_FEATURE_DEBUGGER
    case PROP_DEBUGGER_MANAGER:
      g_value_take_object (value, foundry_context_dup_debugger_manager (self));
      break;
#endif

    case PROP_DEPENDENCY_MANAGER:
      g_value_take_object (value, foundry_context_dup_dependency_manager (self));
      break;

    case PROP_DEVICE_MANAGER:
      g_value_take_object (value, foundry_context_dup_device_manager (self));
      break;

    case PROP_DIAGNOSTIC_MANAGER:
      g_value_take_object (value, foundry_context_dup_diagnostic_manager (self));
      break;

#ifdef FOUNDRY_FEATURE_DOCS
    case PROP_DOCUMENTATION_MANAGER:
      g_value_take_object (value, foundry_context_dup_documentation_manager (self));
      break;
#endif

    case PROP_FILE_MANAGER:
      g_value_take_object (value, foundry_context_dup_file_manager (self));
      break;

#ifdef FOUNDRY_FEATURE_LLM
    case PROP_LLM_MANAGER:
      g_value_take_object (value, foundry_context_dup_llm_manager (self));
      break;
#endif

    case PROP_LOG_MANAGER:
      g_value_take_object (value, foundry_context_dup_log_manager (self));
      break;

#ifdef FOUNDRY_FEATURE_LSP
    case PROP_LSP_MANAGER:
      g_value_take_object (value, foundry_context_dup_lsp_manager (self));
      break;
#endif

    case PROP_OPERATION_MANAGER:
      g_value_take_object (value, foundry_context_dup_operation_manager (self));
      break;

    case PROP_PROJECT_DIRECTORY:
      g_value_take_object (value, foundry_context_dup_project_directory (self));
      break;

    case PROP_RUN_MANAGER:
      g_value_take_object (value, foundry_context_dup_run_manager (self));
      break;

    case PROP_SDK_MANAGER:
      g_value_take_object (value, foundry_context_dup_sdk_manager (self));
      break;

    case PROP_SEARCH_MANAGER:
      g_value_take_object (value, foundry_context_dup_search_manager (self));
      break;

    case PROP_STATE_DIRECTORY:
      g_value_take_object (value, foundry_context_dup_state_directory (self));
      break;

    case PROP_TEST_MANAGER:
      g_value_take_object (value, foundry_context_dup_test_manager (self));
      break;

#ifdef FOUNDRY_FEATURE_TEXT
    case PROP_TEXT_MANAGER:
      g_value_take_object (value, foundry_context_dup_text_manager (self));
      break;
#endif

    case PROP_TITLE:
      g_value_take_string (value, foundry_context_dup_title (self));
      break;

    case PROP_TWEAK_MANAGER:
      g_value_take_object (value, foundry_context_dup_tweak_manager (self));
      break;

#ifdef FOUNDRY_FEATURE_VCS
    case PROP_VCS_MANAGER:
      g_value_take_object (value, foundry_context_dup_vcs_manager (self));
      break;
#endif

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_context_set_property (GObject      *object,
                              guint         prop_id,
                              const GValue *value,
                              GParamSpec   *pspec)
{
  FoundryContext *self = FOUNDRY_CONTEXT (object);

  switch (prop_id)
    {
    case PROP_TITLE:
      foundry_context_set_title (self, g_value_get_string (value));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_context_class_init (FoundryContextClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->dispose = foundry_context_dispose;
  object_class->finalize = foundry_context_finalize;
  object_class->get_property = foundry_context_get_property;
  object_class->set_property = foundry_context_set_property;

  properties[PROP_BUILD_MANAGER] =
    g_param_spec_object ("build-manager", NULL, NULL,
                         FOUNDRY_TYPE_BUILD_MANAGER,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_BUILD_SYSTEM] =
    g_param_spec_string ("build-system", NULL, NULL,
                         NULL,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_COMMAND_MANAGER] =
    g_param_spec_object ("command-manager", NULL, NULL,
                         FOUNDRY_TYPE_COMMAND_MANAGER,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_CONFIG_MANAGER] =
    g_param_spec_object ("config-manager", NULL, NULL,
                         FOUNDRY_TYPE_CONFIG_MANAGER,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

#ifdef FOUNDRY_FEATURE_DEBUGGER
  properties[PROP_DEBUGGER_MANAGER] =
    g_param_spec_object ("debugger-manager", NULL, NULL,
                         FOUNDRY_TYPE_DEBUGGER_MANAGER,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));
#endif

  properties[PROP_DEPENDENCY_MANAGER] =
    g_param_spec_object ("dependency-manager", NULL, NULL,
                         FOUNDRY_TYPE_DEPENDENCY_MANAGER,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_DEVICE_MANAGER] =
    g_param_spec_object ("device-manager", NULL, NULL,
                         FOUNDRY_TYPE_DEVICE_MANAGER,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_DIAGNOSTIC_MANAGER] =
    g_param_spec_object ("diagnostic-manager", NULL, NULL,
                         FOUNDRY_TYPE_DIAGNOSTIC_MANAGER,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

#ifdef FOUNDRY_FEATURE_DOCS
  properties[PROP_DOCUMENTATION_MANAGER] =
    g_param_spec_object ("documentation-manager", NULL, NULL,
                         FOUNDRY_TYPE_DOCUMENTATION_MANAGER,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));
#endif

  properties[PROP_FILE_MANAGER] =
    g_param_spec_object ("file-manager", NULL, NULL,
                         FOUNDRY_TYPE_FILE_MANAGER,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

#ifdef FOUNDRY_FEATURE_LLM
  properties[PROP_LLM_MANAGER] =
    g_param_spec_object ("llm-manager", NULL, NULL,
                         FOUNDRY_TYPE_LLM_MANAGER,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));
#endif

  properties[PROP_LOG_MANAGER] =
    g_param_spec_object ("log-manager", NULL, NULL,
                         FOUNDRY_TYPE_LOG_MANAGER,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

#ifdef FOUNDRY_FEATURE_LSP
  properties[PROP_LSP_MANAGER] =
    g_param_spec_object ("lsp-manager", NULL, NULL,
                         FOUNDRY_TYPE_LSP_MANAGER,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));
#endif

  properties[PROP_OPERATION_MANAGER] =
    g_param_spec_object ("operation-manager", NULL, NULL,
                         FOUNDRY_TYPE_OPERATION_MANAGER,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  /**
   * FoundryContext:project-directory:
   *
   * The directory containing the project.
   *
   * This is generally the directory which contains ".git" and ".foundry".
   */
  properties[PROP_PROJECT_DIRECTORY] =
    g_param_spec_object ("project-directory", NULL, NULL,
                         G_TYPE_FILE,
                         (G_PARAM_READABLE | G_PARAM_STATIC_STRINGS));

  properties[PROP_RUN_MANAGER] =
    g_param_spec_object ("run-manager", NULL, NULL,
                         FOUNDRY_TYPE_RUN_MANAGER,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_SDK_MANAGER] =
    g_param_spec_object ("sdk-manager", NULL, NULL,
                         FOUNDRY_TYPE_SDK_MANAGER,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_SEARCH_MANAGER] =
    g_param_spec_object ("search-manager", NULL, NULL,
                         FOUNDRY_TYPE_SEARCH_MANAGER,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  /**
   * FoundryContext:state-directory:
   *
   * The directory of the context, which is typically ".foundry" within
   * the #FoundryContext:project-directory.
   */
  properties[PROP_STATE_DIRECTORY] =
    g_param_spec_object ("state-directory", NULL, NULL,
                         G_TYPE_FILE,
                         (G_PARAM_READABLE | G_PARAM_STATIC_STRINGS));

  properties[PROP_TEST_MANAGER] =
    g_param_spec_object ("test-manager", NULL, NULL,
                         FOUNDRY_TYPE_TEST_MANAGER,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

#ifdef FOUNDRY_FEATURE_TEXT
  properties[PROP_TEXT_MANAGER] =
    g_param_spec_object ("text-manager", NULL, NULL,
                         FOUNDRY_TYPE_TEXT_MANAGER,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));
#endif

  properties[PROP_TWEAK_MANAGER] =
    g_param_spec_object ("tweak-manager", NULL, NULL,
                         FOUNDRY_TYPE_TWEAK_MANAGER,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_TITLE] =
    g_param_spec_string ("title", NULL, NULL,
                         NULL,
                         (G_PARAM_READWRITE |
                          G_PARAM_EXPLICIT_NOTIFY |
                          G_PARAM_STATIC_STRINGS));

#ifdef FOUNDRY_FEATURE_VCS
  properties[PROP_VCS_MANAGER] =
    g_param_spec_object ("vcs-manager", NULL, NULL,
                         FOUNDRY_TYPE_VCS_MANAGER,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));
#endif

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_context_init (FoundryContext *self)
{
  self->link.data = self;
  self->inhibit = dex_promise_new ();
  self->services = g_ptr_array_new_with_free_func (g_object_unref);
  self->log_manager = g_object_new (FOUNDRY_TYPE_LOG_MANAGER,
                                    "context", self,
                                    NULL);
  self->settings = g_hash_table_new_full (g_str_hash,
                                          g_str_equal,
                                          g_free,
                                          g_object_unref);

  g_ptr_array_add (self->services,
                   g_object_ref (self->log_manager));
  g_ptr_array_add (self->services,
                   g_object_new (FOUNDRY_TYPE_DBUS_SERVICE,
                                 "context", self,
                                 NULL));
  g_ptr_array_add (self->services,
                   g_object_new (FOUNDRY_TYPE_BUILD_MANAGER,
                                 "context", self,
                                 NULL));
  g_ptr_array_add (self->services,
                   g_object_new (FOUNDRY_TYPE_COMMAND_MANAGER,
                                 "context", self,
                                 NULL));
  g_ptr_array_add (self->services,
                   g_object_new (FOUNDRY_TYPE_CONFIG_MANAGER,
                                 "context", self,
                                 NULL));
#ifdef FOUNDRY_FEATURE_DEBUGGER
  g_ptr_array_add (self->services,
                   g_object_new (FOUNDRY_TYPE_DEBUGGER_MANAGER,
                                 "context", self,
                                 NULL));
#endif
  g_ptr_array_add (self->services,
                   g_object_new (FOUNDRY_TYPE_DEPENDENCY_MANAGER,
                                 "context", self,
                                 NULL));
#ifdef FOUNDRY_FEATURE_DOCS
  g_ptr_array_add (self->services,
                   g_object_new (FOUNDRY_TYPE_DOCUMENTATION_MANAGER,
                                 "context", self,
                                 NULL));
#endif
  g_ptr_array_add (self->services,
                   g_object_new (FOUNDRY_TYPE_DEVICE_MANAGER,
                                 "context", self,
                                 NULL));
  g_ptr_array_add (self->services,
                   g_object_new (FOUNDRY_TYPE_DIAGNOSTIC_MANAGER,
                                 "context", self,
                                 NULL));
  g_ptr_array_add (self->services,
                   g_object_new (FOUNDRY_TYPE_FILE_MANAGER,
                                 "context", self,
                                 NULL));
#ifdef FOUNDRY_FEATURE_LLM
  g_ptr_array_add (self->services,
                   g_object_new (FOUNDRY_TYPE_LLM_MANAGER,
                                 "context", self,
                                 NULL));
#endif
#ifdef FOUNDRY_FEATURE_LSP
  g_ptr_array_add (self->services,
                   g_object_new (FOUNDRY_TYPE_LSP_MANAGER,
                                 "context", self,
                                 NULL));
#endif
  g_ptr_array_add (self->services,
                   g_object_new (FOUNDRY_TYPE_OPERATION_MANAGER,
                                 "context", self,
                                 NULL));
  g_ptr_array_add (self->services,
                   g_object_new (FOUNDRY_TYPE_RUN_MANAGER,
                                 "context", self,
                                 NULL));
  g_ptr_array_add (self->services,
                   g_object_new (FOUNDRY_TYPE_SDK_MANAGER,
                                 "context", self,
                                 NULL));
  g_ptr_array_add (self->services,
                   g_object_new (FOUNDRY_TYPE_SEARCH_MANAGER,
                                 "context", self,
                                 NULL));
  g_ptr_array_add (self->services,
                   g_object_new (FOUNDRY_TYPE_TEST_MANAGER,
                                 "context", self,
                                 NULL));
#ifdef FOUNDRY_FEATURE_TEXT
  g_ptr_array_add (self->services,
                   g_object_new (FOUNDRY_TYPE_TEXT_MANAGER,
                                 "context", self,
                                 NULL));
#endif
  g_ptr_array_add (self->services,
                   g_object_new (FOUNDRY_TYPE_TWEAK_MANAGER,
                                 "context", self,
                                 NULL));
#ifdef FOUNDRY_FEATURE_VCS
  g_ptr_array_add (self->services,
                   g_object_new (FOUNDRY_TYPE_VCS_MANAGER,
                                 "context", self,
                                 NULL));
#endif

  G_LOCK (all_contexts);
  g_queue_push_head_link (&all_contexts, &self->link);
  G_UNLOCK (all_contexts);
}

static DexFuture *
create_project_dirs (FoundryContext *self)
{
  g_autoptr(GFile) project_dir = NULL;
  g_autoptr(GFile) user_dir = NULL;
  g_autoptr(GFile) tmp_dir = NULL;

  g_assert (FOUNDRY_IS_CONTEXT (self));

  /* Ensure various subdirectories are created */
  project_dir = g_file_get_child (self->state_directory, "project");
  user_dir = g_file_get_child (self->state_directory, "user");
  tmp_dir = g_file_get_child (self->state_directory, "tmp");

  return dex_future_all (dex_file_make_directory (project_dir, 0),
                         dex_file_make_directory (user_dir, 0),
                         dex_file_make_directory (tmp_dir, 0),
                         NULL);
}

static void
foundry_context_notify_build_system (FoundryContext  *self,
                                     const char      *key,
                                     FoundrySettings *settings)
{
  g_assert (FOUNDRY_IS_CONTEXT (self));
  g_assert (key != NULL);
  g_assert (FOUNDRY_IS_SETTINGS (settings));

  g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_BUILD_SYSTEM]);
}

static void
foundry_context_service_added_cb (PeasExtensionSet *set,
                                  PeasPluginInfo   *plugin_info,
                                  GObject          *extension,
                                  gpointer          user_data)
{
  FoundryService *service = (FoundryService *)extension;

  g_assert (PEAS_IS_EXTENSION_SET (set));
  g_assert (PEAS_IS_PLUGIN_INFO (plugin_info));
  g_assert (FOUNDRY_IS_SERVICE (service));

  dex_future_disown (foundry_service_start (service));
}

static void
foundry_context_service_removed_cb (PeasExtensionSet *set,
                                    PeasPluginInfo   *plugin_info,
                                    GObject          *extension,
                                    gpointer          user_data)
{
  FoundryService *service = (FoundryService *)extension;

  g_assert (PEAS_IS_EXTENSION_SET (set));
  g_assert (PEAS_IS_PLUGIN_INFO (plugin_info));
  g_assert (FOUNDRY_IS_SERVICE (service));

  dex_future_disown (foundry_service_stop (service));
}

typedef struct _FoundryContextNew
{
  GFile               *foundry_dir;
  GFile               *project_dir;
  DexCancellable      *cancellable;
  FoundryContextFlags  flags;
} FoundryContextNew;

static void
foundry_context_new_free (FoundryContextNew *state)
{
  g_clear_object (&state->foundry_dir);
  g_clear_object (&state->project_dir);
  dex_clear (&state->cancellable);
  g_free (state);
}

static gboolean
foundry_context_load_fiber (FoundryContext  *self,
                            GError         **error)
{
  g_autoptr(GPtrArray) futures = NULL;
  g_autoptr(GFile) project_settings = NULL;
  g_autoptr(GFile) user_settings = NULL;
  guint n_items;

  g_assert (FOUNDRY_IS_CONTEXT (self));
  g_assert (G_IS_FILE (self->state_directory));

  if (self->project_directory == NULL)
    self->project_directory = g_file_get_parent (self->state_directory);

  dex_await (create_project_dirs (self), NULL);

  /* Setup custom keyfile backend for user/project settings */
  project_settings = g_file_get_child (self->state_directory, "project/settings.keyfile");
  user_settings = g_file_get_child (self->state_directory, "user/settings.keyfile");
  self->project_settings_backend =
    g_keyfile_settings_backend_new (g_file_peek_path (project_settings),
                                    "/app/devsuite/foundry/",
                                    "app.devsuite.foundry");
  self->user_settings_backend =
    g_keyfile_settings_backend_new (g_file_peek_path (user_settings),
                                    "/app/devsuite/foundry/",
                                    "app.devsuite.foundry");

  /* Keep access to project settings for property notifications */
  self->project_settings = foundry_context_load_settings (self, "app.devsuite.foundry.project", NULL);
  g_signal_connect_object (self->project_settings,
                           "changed::build-system",
                           G_CALLBACK (foundry_context_notify_build_system),
                           self,
                           G_CONNECT_SWAPPED);

  /* Create addins object (but don't yet start them) */
  self->service_addins = peas_extension_set_new (peas_engine_get_default (),
                                                 FOUNDRY_TYPE_SERVICE,
                                                 "context", self,
                                                 NULL);
  g_signal_connect (self->service_addins,
                    "extension-added",
                    G_CALLBACK (foundry_context_service_added_cb),
                    self);
  g_signal_connect (self->service_addins,
                    "extension-removed",
                    G_CALLBACK (foundry_context_service_removed_cb),
                    self);

  /* Request that all services start. Some services may depend
   * on ordering which they may achieve by awaiting on the appropriate
   * future of the dependent service.
   */
  futures = g_ptr_array_new_with_free_func (dex_unref);

  for (guint i = 0; i < self->services->len; i++)
    {
      FoundryService *service = g_ptr_array_index (self->services, i);
      g_autoptr(DexFuture) future = foundry_service_start (service);

      if (future == NULL)
        g_critical ("%s does not implement FoundryServiceClass.start()",
                    G_OBJECT_TYPE_NAME (service));
      else
        g_ptr_array_add (futures,
                         dex_future_catch (g_steal_pointer (&future),
                                           foundry_context_log_failure,
                                           g_object_ref (service),
                                           g_object_unref));
    }

  /* Now start all the extensions coming from plug-ins */
  n_items = g_list_model_get_n_items (G_LIST_MODEL (self->service_addins));
  for (guint i = 0; i < n_items; i++)
    {
      g_autoptr(FoundryService) service = g_list_model_get_item (G_LIST_MODEL (self->service_addins), i);
      g_autoptr(DexFuture) future = foundry_service_start (service);

      if (future == NULL)
        g_critical ("%s does not implement FoundryServiceClass.start()",
                    G_OBJECT_TYPE_NAME (service));
      else
        g_ptr_array_add (futures,
                         dex_future_catch (g_steal_pointer (&future),
                                           foundry_context_log_failure,
                                           g_object_ref (service),
                                           g_object_unref));
    }

  if (futures->len > 0)
    dex_await (dex_future_allv ((DexFuture **)futures->pdata, futures->len), NULL);

  return TRUE;
}

static DexFuture *
foundry_context_new_fiber (gpointer data)
{
  FoundryContextNew *state = data;
  g_autoptr(FoundryContext) self = NULL;
  g_autoptr(GError) error = NULL;
  g_autoptr(GFile) project_dir = NULL;
  g_autoptr(GFile) user_dir = NULL;
  g_autoptr(GFile) tmp_dir = NULL;

  g_assert (state != NULL);
  g_assert (G_IS_FILE (state->foundry_dir));

  /* Make sure startup initialization has completed */
  dex_await (foundry_init (), NULL);

  /* Make sure plugins have also been loaded */
  _foundry_init_plugins ();

  if ((state->flags & (FOUNDRY_CONTEXT_FLAGS_CREATE|FOUNDRY_CONTEXT_FLAGS_SHARED)) != 0)
    {
      g_autoptr(GBytes) bytes = NULL;
      gboolean setup_ignore = TRUE;

      if (!dex_await (dex_file_make_directory_with_parents (state->foundry_dir), &error))
        {
          if (!g_error_matches (error, G_IO_ERROR, G_IO_ERROR_EXISTS) ||
              (state->flags & FOUNDRY_CONTEXT_FLAGS_SHARED) == 0)
            return dex_future_new_for_error (g_steal_pointer (&error));

          setup_ignore = FALSE;
        }

      /* Setup default .gitignore for the .foundry dir */
      if (setup_ignore &&
          (bytes = g_resources_lookup_data ("/app/devsuite/foundry/.foundry/.gitignore", 0, NULL)))
        {
          g_autoptr(GFile) gitignore = g_file_get_child (state->foundry_dir, ".gitignore");

          dex_await (dex_file_replace_contents_bytes (gitignore,
                                                      bytes,
                                                      NULL,
                                                      FALSE,
                                                      G_FILE_CREATE_NONE),
                     NULL);
        }
    }

  self = g_object_new (FOUNDRY_TYPE_CONTEXT, NULL);
  self->is_shared = !!(state->flags & FOUNDRY_CONTEXT_FLAGS_SHARED);
  self->state_directory = g_file_dup (state->foundry_dir);
  self->project_directory = state->project_dir ? g_file_dup (state->project_dir) : NULL;

  if (!foundry_context_load_fiber (self, &error))
    return dex_future_new_for_error (g_steal_pointer (&error));

  return dex_future_new_take_object (g_steal_pointer (&self));
}

/**
 * foundry_context_new:
 * @foundry_dir: the ".foundry" directory
 * @project_dir: (nullable): the project root directory
 * @flags: flags for how to create the context
 * @cancellable: (nullable): optional cancellable to use when awaiting
 *   to propagate work cancellation
 *
 * Creates a new context.
 *
 * If @flags has %FOUNDRY_CONTEXT_FLAGS_CREATE set then it will create
 * the ".foundry" directory first.
 *
 * If @project_dir is not set, the current directory is used unless it
 * was previously stored in the context state.
 *
 * Returns: (transfer full) (not nullable): a #DexFuture which will resolve
 *   to a #FoundryContext.
 */
DexFuture *
foundry_context_new (const char          *foundry_dir,
                     const char          *project_dir,
                     FoundryContextFlags  flags,
                     DexCancellable      *cancellable)
{
  FoundryContextNew *state;

  dex_return_error_if_fail (foundry_dir != NULL);
  dex_return_error_if_fail (!cancellable || DEX_IS_CANCELLABLE (cancellable));

  state = g_new0 (FoundryContextNew, 1);
  state->foundry_dir = g_file_new_for_path (foundry_dir);
  state->project_dir = project_dir ? g_file_new_for_path (project_dir) : NULL;
  state->flags = flags;
  state->cancellable = cancellable ? dex_ref (cancellable) : NULL;

  return dex_scheduler_spawn (NULL, 0,
                              foundry_context_new_fiber,
                              state,
                              (GDestroyNotify) foundry_context_new_free);
}

/**
 * foundry_context_new_for_user:
 * @cancellable: (nullable): optional cancellable to use when awaiting
 *   to propagate work cancellation
 *
 * Returns: (transfer full):
 */
DexFuture *
foundry_context_new_for_user (DexCancellable *cancellable)
{
  g_autofree char *foundry_dir = NULL;

  dex_return_error_if_fail (!cancellable || DEX_IS_CANCELLABLE (cancellable));

  foundry_dir = _foundry_get_shared_dir ();

  return foundry_context_new (foundry_dir,
                              g_get_home_dir (),
                              FOUNDRY_CONTEXT_FLAGS_SHARED,
                              cancellable);
}

/**
 * foundry_context_save:
 * @self: a #FoundryContext
 *
 * Save the foundry state to the #FoundryContext:directory.
 *
 * Returns: (transfer full) (not nullable): A #DexFuture that will resolve to
 *   a boolean.
 */
DexFuture *
foundry_context_save (FoundryContext *self,
                      DexCancellable *cancellable)
{
  g_return_val_if_fail (FOUNDRY_IS_CONTEXT (self), NULL);
  g_return_val_if_fail (!cancellable || DEX_IS_CANCELLABLE (cancellable), NULL);

  return dex_future_new_true ();
}

/**
 * foundry_context_dup_project_directory:
 * @self: a #FoundryContext
 *
 * Gets the #FoundryContext:project-directory.
 *
 * Returns: (transfer full) (not nullable): a #GFile
 */
GFile *
foundry_context_dup_project_directory (FoundryContext *self)
{
  g_return_val_if_fail (FOUNDRY_IS_CONTEXT (self), NULL);

  return g_object_ref (self->project_directory);
}

/**
 * foundry_context_dup_state_directory:
 * @self: a #FoundryContext
 *
 * Gets the #FoundryContext:state-directory.
 *
 * Returns: (transfer full) (not nullable): a #GFile
 */
GFile *
foundry_context_dup_state_directory (FoundryContext *self)
{
  g_return_val_if_fail (FOUNDRY_IS_CONTEXT (self), NULL);

  return g_object_ref (self->state_directory);
}

typedef struct _FoundryContetDiscover
{
  char           *path;
  DexCancellable *cancellable;
} FoundryContextDiscover;

static void
foundry_context_discover_free (FoundryContextDiscover *state)
{
  g_clear_pointer (&state->path, g_free);
  dex_clear (&state->cancellable);
  g_free (state);
}

static DexFuture *
foundry_context_discover_fiber (gpointer data)
{
  FoundryContextDiscover *state = data;
  g_autoptr(GFile) file = NULL;

  g_assert (state != NULL);
  g_assert (state->path != NULL);

  file = g_file_new_for_path (state->path);

  while (file != NULL)
    {
      g_autofree char *name = g_file_get_basename (file);
      g_autoptr(GFile) child = g_file_get_child (file, ".foundry");
      g_autoptr(GFile) parent = NULL;
      g_autoptr(GError) error = NULL;
      g_autoptr(DexFuture) query = NULL;

      if (g_str_equal (name, ".foundry"))
        return dex_future_new_take_string (g_file_get_path (file));

      query = dex_file_query_exists (child);

      if (!dex_await (dex_future_first (dex_ref (state->cancellable),
                                        dex_ref (query),
                                        NULL),
                      &error))
        {
          if (g_error_matches (error, G_IO_ERROR, G_IO_ERROR_CANCELLED))
            return dex_future_new_for_error (g_steal_pointer (&error));
        }

      if (dex_await (g_steal_pointer (&query), NULL))
        return dex_future_new_take_string (g_file_get_path (child));

      parent = g_file_get_parent (file);
      g_set_object (&file, parent);
    }

  return dex_future_new_reject (G_IO_ERROR,
                                G_IO_ERROR_NOT_FOUND,
                                "Failed to locate '.foundry' directory for '%s'",
                                state->path);
}

/**
 * foundry_context_discover:
 * @path: the starting path
 * @cancellable: (nullable): an optional cancellable
 *
 * Attempts to locate the nearest .foundry directory starting from @path.
 *
 * Returns: (transfer full): a #DexFuture that resolves to a path in the
 *   file system encoding.
 */
DexFuture *
foundry_context_discover (const char     *path,
                          DexCancellable *cancellable)
{
  FoundryContextDiscover *state;

  g_return_val_if_fail (path != NULL, NULL);
  g_return_val_if_fail (!cancellable || DEX_IS_CANCELLABLE (cancellable), NULL);

  state = g_new0 (FoundryContextDiscover, 1);
  state->path = g_strdup (path);
  state->cancellable = cancellable ? dex_ref (cancellable) : dex_cancellable_new ();

  return dex_scheduler_spawn (NULL, 0,
                              foundry_context_discover_fiber,
                              state,
                              (GDestroyNotify) foundry_context_discover_free);
}

static DexFuture *
foundry_context_shutdown_fiber (gpointer user_data)
{
  FoundryContext *self = user_data;
  g_autoptr(GPtrArray) futures = NULL;
  guint n_items;

  g_assert (FOUNDRY_IS_CONTEXT (self));

  /* First wait for our inhibit count to reach zero */
  if (self->inhibit_count > 0)
    {
      g_assert (DEX_IS_FUTURE (self->inhibit));
      dex_await (dex_ref (DEX_FUTURE (self->inhibit)), NULL);
    }

  g_signal_handlers_disconnect_by_func (self->service_addins,
                                        G_CALLBACK (foundry_context_service_added_cb),
                                        self);
  g_signal_handlers_disconnect_by_func (self->service_addins,
                                        G_CALLBACK (foundry_context_service_removed_cb),
                                        self);

  /* Request that all services shutdown. Some services may depend
   * on ordering which they may achieve by awaiting on the appropriate
   * future of the dependent service.
   */
  futures = g_ptr_array_new_with_free_func (dex_unref);

  /* Call stop on addin services first */
  n_items = g_list_model_get_n_items (G_LIST_MODEL (self->service_addins));
  for (guint i = 0; i < n_items; i++)
    {
      g_autoptr(FoundryService) service = g_list_model_get_item (G_LIST_MODEL (self->service_addins), i);
      g_autoptr(DexFuture) future = foundry_service_stop (service);

      if (future == NULL)
        g_critical ("%s does not implement FoundryServiceClass.stop()",
                    G_OBJECT_TYPE_NAME (service));
      else
        g_ptr_array_add (futures,
                         dex_future_catch (g_steal_pointer (&future),
                                           foundry_context_log_failure,
                                           g_object_ref (service),
                                           g_object_unref));
    }

  g_clear_object (&self->service_addins);

  /* Now handle core services */
  for (guint i = 0; i < self->services->len; i++)
    {
      FoundryService *service = g_ptr_array_index (self->services, i);
      g_autoptr(DexFuture) future = foundry_service_stop (service);

      if (future == NULL)
        g_critical ("%s does not implement FoundryServiceClass.stop()",
                    G_OBJECT_TYPE_NAME (service));
      else
        g_ptr_array_add (futures,
                         dex_future_catch (g_steal_pointer (&future),
                                           foundry_context_log_failure,
                                           g_object_ref (service),
                                           g_object_unref));
    }

  dex_await (dex_future_allv ((DexFuture **)futures->pdata, futures->len), NULL);

  return dex_future_new_true ();
}

/**
 * foundry_context_shutdown:
 * @self: a #FoundryContext
 *
 * Requests that the context shutdown and cleanup state.
 *
 * Returns: (transfer full): a #DexFuture that resolves when the
 *  context has shutdown.
 */
DexFuture *
foundry_context_shutdown (FoundryContext *self)
{
  g_return_val_if_fail (FOUNDRY_IS_CONTEXT (self), NULL);

  if (self->shutdown == NULL)
    self->shutdown = dex_scheduler_spawn (NULL, 0,
                                          foundry_context_shutdown_fiber,
                                          g_object_ref (self),
                                          g_object_unref);

  return dex_ref (self->shutdown);
}

/**
 * foundry_context_dup_service_typed:
 * @self: a [class@Foundry.Context]
 *
 * Returns: (transfer full) (type FoundryService):
 */
gpointer
foundry_context_dup_service_typed (FoundryContext *self,
                                   GType           type)
{
  guint n_items;

  g_return_val_if_fail (FOUNDRY_IS_CONTEXT (self), NULL);
  g_return_val_if_fail (type != FOUNDRY_TYPE_SERVICE &&
                        g_type_is_a (type, FOUNDRY_TYPE_SERVICE),
                        NULL);

  for (guint i = 0; i < self->services->len; i++)
    {
      FoundryService *service = g_ptr_array_index (self->services, i);

      if (g_type_is_a (G_OBJECT_TYPE (service), type))
        return g_object_ref (service);
    }

  if (self->service_addins == NULL)
    return NULL;

  n_items = g_list_model_get_n_items (G_LIST_MODEL (self->service_addins));

  for (guint i = 0; i < n_items; i++)
    {
      g_autoptr(FoundryService) service = g_list_model_get_item (G_LIST_MODEL (self->service_addins), i);

      if (g_type_is_a (G_OBJECT_TYPE (service), type))
        return g_steal_pointer (&service);
    }

  return NULL;
}

/**
 * foundry_context_dup_dbus_service:
 * @self: a #FoundryContext
 *
 * Gets the #FoundryDBusService instance.
 *
 * Returns: (transfer full): a #FoundryDBusService
 */
FoundryDBusService *
foundry_context_dup_dbus_service (FoundryContext *self)
{
  g_return_val_if_fail (FOUNDRY_IS_CONTEXT (self), NULL);

  return foundry_context_dup_service_typed (self, FOUNDRY_TYPE_DBUS_SERVICE);
}

/**
 * foundry_context_dup_build_manager:
 * @self: a #FoundryContext
 *
 * Gets the #FoundryBuildManager instance.
 *
 * Returns: (transfer full): a #FoundryBuildManager
 */
FoundryBuildManager *
foundry_context_dup_build_manager (FoundryContext *self)
{
  g_return_val_if_fail (FOUNDRY_IS_CONTEXT (self), NULL);

  return foundry_context_dup_service_typed (self, FOUNDRY_TYPE_BUILD_MANAGER);
}

/**
 * foundry_context_dup_command_manager:
 * @self: a #FoundryContext
 *
 * Gets the #FoundryCommandManager instance.
 *
 * Returns: (transfer full): a #FoundryCommandManager
 */
FoundryCommandManager *
foundry_context_dup_command_manager (FoundryContext *self)
{
  g_return_val_if_fail (FOUNDRY_IS_CONTEXT (self), NULL);

  return foundry_context_dup_service_typed (self, FOUNDRY_TYPE_COMMAND_MANAGER);
}

/**
 * foundry_context_dup_config_manager:
 * @self: a #FoundryContext
 *
 * Gets the #FoundryConfigManager instance.
 *
 * Returns: (transfer full): a #FoundryConfigManager
 */
FoundryConfigManager *
foundry_context_dup_config_manager (FoundryContext *self)
{
  g_return_val_if_fail (FOUNDRY_IS_CONTEXT (self), NULL);

  return foundry_context_dup_service_typed (self, FOUNDRY_TYPE_CONFIG_MANAGER);
}

/**
 * foundry_context_dup_dependency_manager:
 * @self: a #FoundryContext
 *
 * Gets the #FoundryDependencyManager instance.
 *
 * Returns: (transfer full): a #FoundryDependencyManager
 */
FoundryDependencyManager *
foundry_context_dup_dependency_manager (FoundryContext *self)
{
  g_return_val_if_fail (FOUNDRY_IS_CONTEXT (self), NULL);

  return foundry_context_dup_service_typed (self, FOUNDRY_TYPE_DEPENDENCY_MANAGER);
}

/**
 * foundry_context_dup_device_manager:
 * @self: a #FoundryContext
 *
 * Gets the #FoundryDeviceManager instance.
 *
 * Returns: (transfer full): a #FoundryDeviceManager
 */
FoundryDeviceManager *
foundry_context_dup_device_manager (FoundryContext *self)
{
  g_return_val_if_fail (FOUNDRY_IS_CONTEXT (self), NULL);

  return foundry_context_dup_service_typed (self, FOUNDRY_TYPE_DEVICE_MANAGER);
}

/**
 * foundry_context_dup_diagnostic_manager:
 * @self: a #FoundryContext
 *
 * Gets the #FoundryDiagnosticManager instance.
 *
 * Returns: (transfer full): a #FoundryDiagnosticManager
 */
FoundryDiagnosticManager *
foundry_context_dup_diagnostic_manager (FoundryContext *self)
{
  g_return_val_if_fail (FOUNDRY_IS_CONTEXT (self), NULL);

  return foundry_context_dup_service_typed (self, FOUNDRY_TYPE_DIAGNOSTIC_MANAGER);
}

#ifdef FOUNDRY_FEATURE_DOCS
/**
 * foundry_context_dup_documentation_manager:
 * @self: a #FoundryContext
 *
 * Gets the #FoundryDocumentationManager instance.
 *
 * Returns: (transfer full): a #FoundryDocumentationManager
 */
FoundryDocumentationManager *
foundry_context_dup_documentation_manager (FoundryContext *self)
{
  g_return_val_if_fail (FOUNDRY_IS_CONTEXT (self), NULL);

  return foundry_context_dup_service_typed (self, FOUNDRY_TYPE_DOCUMENTATION_MANAGER);
}
#endif

/**
 * foundry_context_dup_file_manager:
 * @self: a #FoundryContext
 *
 * Gets the #FoundryFileManager instance.
 *
 * Returns: (transfer full): a #FoundryFileManager
 */
FoundryFileManager *
foundry_context_dup_file_manager (FoundryContext *self)
{
  g_return_val_if_fail (FOUNDRY_IS_CONTEXT (self), NULL);

  return foundry_context_dup_service_typed (self, FOUNDRY_TYPE_FILE_MANAGER);
}

#ifdef FOUNDRY_FEATURE_DEBUGGER
/**
 * foundry_context_dup_debugger_manager:
 * @self: a #FoundryContext
 *
 * Gets the #FoundryDebuggerManager instance.
 *
 * Returns: (transfer full): a #FoundryDebuggerManager
 */
FoundryDebuggerManager *
foundry_context_dup_debugger_manager (FoundryContext *self)
{
  g_return_val_if_fail (FOUNDRY_IS_CONTEXT (self), NULL);

  return foundry_context_dup_service_typed (self, FOUNDRY_TYPE_DEBUGGER_MANAGER);
}
#endif

#ifdef FOUNDRY_FEATURE_LLM
/**
 * foundry_context_dup_llm_manager:
 * @self: a #FoundryContext
 *
 * Gets the #FoundryLlmManager instance.
 *
 * Returns: (transfer full): a #FoundryLlmManager
 */
FoundryLlmManager *
foundry_context_dup_llm_manager (FoundryContext *self)
{
  g_return_val_if_fail (FOUNDRY_IS_CONTEXT (self), NULL);

  return foundry_context_dup_service_typed (self, FOUNDRY_TYPE_LLM_MANAGER);
}
#endif

/**
 * foundry_context_dup_log_manager:
 * @self: a #FoundryContext
 *
 * Gets the #FoundryLogManager instance.
 *
 * Returns: (transfer full): a #FoundryLogManager
 */
FoundryLogManager *
foundry_context_dup_log_manager (FoundryContext *self)
{
  g_return_val_if_fail (FOUNDRY_IS_CONTEXT (self), NULL);

  return foundry_context_dup_service_typed (self, FOUNDRY_TYPE_LOG_MANAGER);
}

#ifdef FOUNDRY_FEATURE_LSP
/**
 * foundry_context_dup_lsp_manager:
 * @self: a #FoundryContext
 *
 * Gets the #FoundryLspManager instance.
 *
 * Returns: (transfer full): a #FoundryLspManager
 */
FoundryLspManager *
foundry_context_dup_lsp_manager (FoundryContext *self)
{
  g_return_val_if_fail (FOUNDRY_IS_CONTEXT (self), NULL);

  return foundry_context_dup_service_typed (self, FOUNDRY_TYPE_LSP_MANAGER);
}
#endif

/**
 * foundry_context_dup_operation_manager:
 * @self: a #FoundryContext
 *
 * Gets the #FoundryOperationManager instance.
 *
 * Returns: (transfer full): a #FoundryOperationManager
 */
FoundryOperationManager *
foundry_context_dup_operation_manager (FoundryContext *self)
{
  g_return_val_if_fail (FOUNDRY_IS_CONTEXT (self), NULL);

  return foundry_context_dup_service_typed (self, FOUNDRY_TYPE_OPERATION_MANAGER);
}

/**
 * foundry_context_dup_run_manager:
 * @self: a #FoundryContext
 *
 * Gets the #FoundryRunManager instance.
 *
 * Returns: (transfer full): a #FoundryRunManager
 */
FoundryRunManager *
foundry_context_dup_run_manager (FoundryContext *self)
{
  g_return_val_if_fail (FOUNDRY_IS_CONTEXT (self), NULL);

  return foundry_context_dup_service_typed (self, FOUNDRY_TYPE_RUN_MANAGER);
}

/**
 * foundry_context_dup_sdk_manager:
 * @self: a #FoundryContext
 *
 * Gets the #FoundrySdkManager instance.
 *
 * Returns: (transfer full): a #FoundrySdkManager
 */
FoundrySdkManager *
foundry_context_dup_sdk_manager (FoundryContext *self)
{
  g_return_val_if_fail (FOUNDRY_IS_CONTEXT (self), NULL);

  return foundry_context_dup_service_typed (self, FOUNDRY_TYPE_SDK_MANAGER);
}

/**
 * foundry_context_dup_test_manager:
 * @self: a #FoundryContext
 *
 * Gets the #FoundryTestManager instance.
 *
 * Returns: (transfer full): a #FoundryTestManager
 */
FoundryTestManager *
foundry_context_dup_test_manager (FoundryContext *self)
{
  g_return_val_if_fail (FOUNDRY_IS_CONTEXT (self), NULL);

  return foundry_context_dup_service_typed (self, FOUNDRY_TYPE_TEST_MANAGER);
}

#ifdef FOUNDRY_FEATURE_TEXT
/**
 * foundry_context_dup_text_manager:
 * @self: a #FoundryContext
 *
 * Gets the #FoundryTextManager instance.
 *
 * Returns: (transfer full): a #FoundryTextManager
 */
FoundryTextManager *
foundry_context_dup_text_manager (FoundryContext *self)
{
  g_return_val_if_fail (FOUNDRY_IS_CONTEXT (self), NULL);

  return foundry_context_dup_service_typed (self, FOUNDRY_TYPE_TEXT_MANAGER);
}
#endif

/**
 * foundry_context_dup_tweak_manager:
 * @self: a #FoundryContext
 *
 * Gets the #FoundryTweakManager instance.
 *
 * Returns: (transfer full): a #FoundryTweakManager
 */
FoundryTweakManager *
foundry_context_dup_tweak_manager (FoundryContext *self)
{
  g_return_val_if_fail (FOUNDRY_IS_CONTEXT (self), NULL);

  return foundry_context_dup_service_typed (self, FOUNDRY_TYPE_TWEAK_MANAGER);
}

#ifdef FOUNDRY_FEATURE_VCS
/**
 * foundry_context_dup_vcs_manager:
 * @self: a #FoundryContext
 *
 * Gets the #FoundryVcsManager instance.
 *
 * Returns: (transfer full): a #FoundryVcsManager
 */
FoundryVcsManager *
foundry_context_dup_vcs_manager (FoundryContext *self)
{
  g_return_val_if_fail (FOUNDRY_IS_CONTEXT (self), NULL);

  return foundry_context_dup_service_typed (self, FOUNDRY_TYPE_VCS_MANAGER);
}
#endif

/**
 * foundry_context_dup_search_manager:
 * @self: a #FoundryContext
 *
 * Gets the #FoundrySearchManager instance.
 *
 * Returns: (transfer full): a #FoundrySearchManager
 */
FoundrySearchManager *
foundry_context_dup_search_manager (FoundryContext *self)
{
  g_return_val_if_fail (FOUNDRY_IS_CONTEXT (self), NULL);

  return foundry_context_dup_service_typed (self, FOUNDRY_TYPE_SEARCH_MANAGER);
}

DexFuture *
_foundry_context_shutdown_all (void)
{
  g_autoptr(GPtrArray) futures = g_ptr_array_new_with_free_func (dex_unref);

  G_LOCK (all_contexts);

  for (const GList *iter = all_contexts.head; iter; iter = iter->next)
    {
      FoundryContext *context = iter->data;
      g_assert (FOUNDRY_IS_CONTEXT (context));
      g_ptr_array_add (futures, foundry_context_shutdown (context));
    }

  G_UNLOCK (all_contexts);

  if (futures->len == 0)
    return dex_future_new_true ();

  return dex_future_allv ((DexFuture **)futures->pdata, futures->len);
}

void
foundry_context_logv (FoundryContext *self,
                      const char     *domain,
                      GLogLevelFlags  severity,
                      const char     *format,
                      va_list         args)
{
  g_autofree char *message = NULL;

  g_return_if_fail (!self || FOUNDRY_IS_CONTEXT (self));

#ifdef __clang__
# pragma clang diagnostic push
# pragma clang diagnostic ignored "-Wformat-nonliteral"
#elif G_GNUC_CHECK_VERSION(4,0)
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wsuggest-attribute=format"
#endif

  if (!FOUNDRY_IS_CONTEXT (self) || self->log_manager == NULL)
    {
      g_logv (domain, severity, format, args);
      return;
    }

  message = g_strdup_vprintf (format, args);

  _foundry_log_manager_append (self->log_manager,
                               domain,
                               severity,
                               g_steal_pointer (&message));

#ifdef __clang__
# pragma clang diagnostic pop
#elif G_GNUC_CHECK_VERSION(4,0)
# pragma GCC diagnostic pop
#endif
}

void
foundry_context_log (FoundryContext *self,
                     const char     *domain,
                     GLogLevelFlags  severity,
                     const char     *format,
                     ...)
{
  va_list args;

  g_return_if_fail (!self || FOUNDRY_IS_CONTEXT (self));

  va_start (args, format);
  foundry_context_logv (self, domain, severity, format, args);
  va_end (args);
}

/**
 * foundry_context_load_settings:
 * @self: a #FoundryContext
 * @schema_id: the gsettings schema identifier
 * @schema_path: (nullable): an optional schema path
 *
 * Loads layered [class@Gio.Settings] as a [class@Foundry.Settings].
 *
 * The [class@Foundry.Settings] allows for settings to come from multiple
 * layers such as user-defaults, project-defaults, and user-overrides.
 *
 * Returns: (transfer full): a [class@Foundry.Settings]
 *
 */
FoundrySettings *
foundry_context_load_settings (FoundryContext *self,
                               const char     *schema_id,
                               const char     *schema_path)
{
  g_autofree char *key = NULL;
  FoundrySettings *settings;

  g_return_val_if_fail (FOUNDRY_IS_CONTEXT (self), NULL);
  g_return_val_if_fail (schema_id != NULL, NULL);

  /* We do not cache these or share them because GSettings objects
   * are intended to be cheap (while the backend is more complex).
   *
   * Additionally, you can only really bind one of them to various
   * UI components nor are they thread-safe. Best to hand out a new
   * one each time and rely on the backend for synchronization.
   */

  if (schema_path == NULL)
    key = g_strdup_printf ("%s:", schema_id);
  else
    key = g_strdup_printf ("%s:%s", schema_id, schema_path);

  if (!(settings = g_hash_table_lookup (self->settings, key)))
    {
      settings = foundry_settings_new_with_path (self, schema_id, schema_path);
      g_hash_table_replace (self->settings, g_steal_pointer (&key), settings);
    }

  return g_object_ref (settings);
}

GSettingsBackend *
_foundry_context_dup_project_settings_backend (FoundryContext *self)
{
  g_return_val_if_fail (FOUNDRY_IS_CONTEXT (self), NULL);

  return g_object_ref (self->project_settings_backend);
}

GSettingsBackend *
_foundry_context_dup_user_settings_backend (FoundryContext *self)
{
  g_return_val_if_fail (FOUNDRY_IS_CONTEXT (self), NULL);

  return g_object_ref (self->user_settings_backend);
}

/**
 * foundry_context_load_project_settings:
 * @self: a #FoundryContext
 *
 * This function is functionally equivalent to calling
 * [method@Foundry.Context.load_settings] with the "app.devsuite.foundry.project"
 * gsettings schema id.
 *
 * Returns: (transfer full): a [class@Foundry.Settings]
 */
FoundrySettings *
foundry_context_load_project_settings (FoundryContext *self)
{
  g_return_val_if_fail (FOUNDRY_IS_CONTEXT (self), NULL);

  return foundry_context_load_settings (self, "app.devsuite.foundry.project", NULL);
}

/**
 * foundry_context_network_allowed:
 * @self: a [class@Foundry.Context]
 *
 * Checks if network is currently allowed.
 *
 * This checks if data is allowed on metered connections and if the current
 * network connection is metered.
 *
 * Returns: %TRUE if network is allowed
 */
gboolean
foundry_context_network_allowed (FoundryContext *self)
{
  g_autoptr(FoundrySettings) settings = NULL;
  GNetworkMonitor *monitor;

  g_return_val_if_fail (FOUNDRY_IS_CONTEXT (self), FALSE);

  settings = foundry_context_load_settings (self, "app.devsuite.foundry.network", NULL);
  if (foundry_settings_get_boolean (settings, "allow-when-metered"))
    return TRUE;

  monitor = g_network_monitor_get_default ();
  return !g_network_monitor_get_network_metered (monitor);
}

static GFile *
foundry_context_dup_cache_root (FoundryContext *self)
{
  g_assert (FOUNDRY_IS_CONTEXT (self));

  return g_file_get_child (self->state_directory, "cache");
}

/**
 * foundry_context_cache_filename:
 * @self: a [class@Foundry.Context]
 *
 * Returns: (transfer full): a new path to the file
 */
char *
foundry_context_cache_filename (FoundryContext *self,
                                ...)
{
  g_autoptr(GFile) cache_root = NULL;
  g_autofree char *path = NULL;
  va_list args;

  g_return_val_if_fail (FOUNDRY_IS_CONTEXT (self), NULL);

  cache_root = foundry_context_dup_cache_root (self);

  va_start (args, self);
  path = g_build_filename_valist (g_file_peek_path (cache_root), &args);
  va_end (args);

  return g_steal_pointer (&path);
}

/**
 * foundry_context_cache_file:
 * @self: a [class@Foundry.Context]
 *
 * Returns: (transfer full): a [iface@Gio.File] within the cache root
 */
GFile *
foundry_context_cache_file (FoundryContext *self,
                            ...)
{
  g_autoptr(GFile) cache_root = NULL;
  g_autofree char *path = NULL;
  va_list args;

  g_return_val_if_fail (FOUNDRY_IS_CONTEXT (self), NULL);

  cache_root = foundry_context_dup_cache_root (self);

  va_start (args, self);
  path = g_build_filename_valist (g_file_peek_path (cache_root), &args);
  va_end (args);

  return g_file_new_for_path (path);
}

/**
 * foundry_context_tmp_filename:
 * @self: a [class@Foundry.Context]
 *
 * Returns a path that will be in the "tmp/" directory of the .foundry dir.
 *
 * Returns: (transfer full): a new path to the file
 *
 * Since: 1.1
 */
char *
foundry_context_tmp_filename (FoundryContext *self,
                              ...)
{
  g_autoptr(GFile) tmp_root = NULL;
  g_autofree char *path = NULL;
  va_list args;

  g_return_val_if_fail (FOUNDRY_IS_CONTEXT (self), NULL);

  tmp_root = g_file_get_child (self->state_directory, "tmp");

  va_start (args, self);
  path = g_build_filename_valist (g_file_peek_path (tmp_root), &args);
  va_end (args);

  return g_steal_pointer (&path);
}

/**
 * foundry_context_dup_build_system:
 * @self: a [class@Foundry.Context]
 *
 * Gets the name of the build system to use.
 *
 * First the settings are checked. If set, that is preferred. After that,
 * the configuration is checked to see if it specifies a build system.
 *
 * Otherwise, %NULL is returned.
 *
 * Returns: (transfer full) (nullable): a build-system name or %NULL
 */
char *
foundry_context_dup_build_system (FoundryContext *self)
{
  g_autoptr(FoundryConfigManager) config_manager = NULL;
  g_autoptr(FoundrySettings) settings = NULL;
  g_autoptr(FoundryConfig) config = NULL;
  g_autofree char *build_system = NULL;

  g_return_val_if_fail (FOUNDRY_IS_CONTEXT (self), NULL);

  settings = foundry_context_load_settings (self, "app.devsuite.foundry.project", NULL);
  build_system = foundry_settings_get_string (settings, "build-system");

  if (!foundry_str_empty0 (build_system))
    return g_steal_pointer (&build_system);

  config_manager = foundry_context_dup_config_manager (self);
  config = foundry_config_manager_dup_config (config_manager);

  if (config != NULL)
    return foundry_config_dup_build_system (config);

  return NULL;
}

gboolean
_foundry_context_inhibit (FoundryContext *self)
{
  g_return_val_if_fail (FOUNDRY_IS_MAIN_THREAD (), FALSE);
  g_return_val_if_fail (FOUNDRY_IS_CONTEXT (self), FALSE);

  if (foundry_context_in_shutdown (self))
    return FALSE;

  self->inhibit_count++;

  return TRUE;
}

void
_foundry_context_uninhibit (FoundryContext *self)
{
  g_return_if_fail (FOUNDRY_IS_MAIN_THREAD ());
  g_return_if_fail (FOUNDRY_IS_CONTEXT (self));
  g_return_if_fail (self->inhibit_count > 0);

  self->inhibit_count--;

  if (self->inhibit_count == 0 && self->shutdown != NULL)
    dex_promise_resolve_boolean (self->inhibit, TRUE);
}

gboolean
foundry_context_is_shared (FoundryContext *self)
{
  g_return_val_if_fail (FOUNDRY_IS_CONTEXT (self), FALSE);

  return self->is_shared;
}

/**
 * foundry_context_dup_title:
 * @self: a [class@Foundry.Context]
 *
 * Returns: (transfer full) (nullable):
 */
char *
foundry_context_dup_title (FoundryContext *self)
{
  g_return_val_if_fail (FOUNDRY_IS_CONTEXT (self), NULL);

  return g_strdup (self->title);
}

void
foundry_context_set_title (FoundryContext *self,
                           const char     *title)
{
  g_return_if_fail (FOUNDRY_IS_MAIN_THREAD ());
  g_return_if_fail (FOUNDRY_IS_CONTEXT (self));

  if (g_set_str (&self->title, title))
    g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_TITLE]);
}

static const char *
get_action_prefix (FoundryService *service)
{
  const char *prefix = foundry_service_class_get_action_prefix (FOUNDRY_SERVICE_GET_CLASS (service));

  if (prefix == NULL)
    return G_OBJECT_TYPE_NAME (service);

  return prefix;
}

static void
foundry_context_action_group_service_added_cb (PeasExtensionSet   *set,
                                               PeasPluginInfo     *plugin_info,
                                               FoundryService     *service,
                                               FoundryActionMuxer *muxer)
{
  GActionGroup *action_group;

  g_assert (PEAS_IS_EXTENSION_SET (set));
  g_assert (PEAS_IS_PLUGIN_INFO (plugin_info));
  g_assert (FOUNDRY_IS_SERVICE (service));
  g_assert (FOUNDRY_IS_ACTION_MUXER (muxer));

  action_group = foundry_service_get_action_group (service);

  g_assert (!action_group || G_IS_ACTION_GROUP (action_group));

  if (action_group != NULL)
    foundry_action_muxer_insert_action_group (muxer,
                                              get_action_prefix (service),
                                              action_group);
}

static void
foundry_context_action_group_service_removed_cb (PeasExtensionSet   *set,
                                                 PeasPluginInfo     *plugin_info,
                                                 FoundryService     *service,
                                                 FoundryActionMuxer *muxer)
{
  g_assert (PEAS_IS_EXTENSION_SET (set));
  g_assert (PEAS_IS_PLUGIN_INFO (plugin_info));
  g_assert (FOUNDRY_IS_SERVICE (service));
  g_assert (FOUNDRY_IS_ACTION_MUXER (muxer));

  foundry_action_muxer_remove_action_group (muxer, get_action_prefix (service));
}

/**
 * foundry_context_dup_action_group:
 * @self: a [class@Foundry.Context]
 *
 * Gets a [iface@Gio.ActionGroup] that contains various actions for the context.
 *
 * Actions may be provided by subclassing FoundryService and implementing the
 * [iface@Gio.ActionGroup] interface.
 *
 * Returns: (transfer full): a [iface@Gio.ActionGroup]
 */
GActionGroup *
foundry_context_dup_action_group (FoundryContext *self)
{
  g_autoptr(FoundryActionMuxer) muxer = NULL;
  guint n_items;

  g_return_val_if_fail (FOUNDRY_IS_CONTEXT (self), NULL);

  muxer = foundry_action_muxer_new ();

  for (guint i = 0; i < self->services->len; i++)
    {
      FoundryService *service = g_ptr_array_index (self->services, i);
      GActionGroup *action_group = foundry_service_get_action_group (service);

      g_assert (!action_group || G_IS_ACTION_GROUP (action_group));

      if (action_group != NULL)
        foundry_action_muxer_insert_action_group (muxer,
                                                  get_action_prefix (service),
                                                  action_group);
    }

  n_items = g_list_model_get_n_items (G_LIST_MODEL (self->service_addins));

  for (guint i = 0; i < n_items; i++)
    {
      g_autoptr(FoundryService) service = g_list_model_get_item (G_LIST_MODEL (self->service_addins), i);
      GActionGroup *action_group = foundry_service_get_action_group (service);

      g_assert (!action_group || G_IS_ACTION_GROUP (action_group));

      if (action_group != NULL)
        foundry_action_muxer_insert_action_group (muxer,
                                                  get_action_prefix (service),
                                                  action_group);
    }

  g_signal_connect_object (self->service_addins,
                           "extension-added",
                           G_CALLBACK (foundry_context_action_group_service_added_cb),
                           muxer,
                           0);

  g_signal_connect_object (self->service_addins,
                           "extension-removed",
                           G_CALLBACK (foundry_context_action_group_service_removed_cb),
                           muxer,
                           0);

  return G_ACTION_GROUP (g_steal_pointer (&muxer));
}

/**
 * foundry_context_list_services:
 * @self: a [class@Foundry.Context]
 *
 * Returns: (transfer full): a [iface@Gio.ListModel] of [class@Foundry.Service]
 */
GListModel *
foundry_context_list_services (FoundryContext *self)
{
  GListStore *store;

  g_return_val_if_fail (FOUNDRY_IS_CONTEXT (self), NULL);

  store = g_list_store_new (FOUNDRY_TYPE_SERVICE);
  g_list_store_splice (store, 0, 0, self->services->pdata, self->services->len);
  return G_LIST_MODEL (store);
}

/**
 * foundry_context_dup_default_license:
 * @self: a [class@Foundry.Context]
 *
 * Returns: (transfer full) (nullable):
 */
FoundryLicense *
foundry_context_dup_default_license (FoundryContext *self)
{
  g_autoptr(FoundrySettings) settings = NULL;
  g_autofree char *default_license = NULL;

  g_return_val_if_fail (FOUNDRY_IS_CONTEXT (self), NULL);

  settings = foundry_context_load_project_settings (self);
  default_license = foundry_settings_get_string (settings, "default-license");

  if (default_license == NULL || default_license[0] == 0)
    return NULL;

  return foundry_license_find (default_license);
}
