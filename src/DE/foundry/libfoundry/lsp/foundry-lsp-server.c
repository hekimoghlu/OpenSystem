/* foundry-lsp-server.c
 *
 * Copyright 2025 Christian Hergert <chergert@redhat.com>
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

#include "foundry-build-pipeline.h"
#include "foundry-lsp-server.h"
#include "foundry-process-launcher.h"

G_DEFINE_ABSTRACT_TYPE (FoundryLspServer, foundry_lsp_server, FOUNDRY_TYPE_CONTEXTUAL)

enum {
  PROP_0,
  PROP_NAME,
  PROP_LANGUAGES,
  N_PROPS,
};

static GParamSpec *properties[N_PROPS];

static gboolean
foundry_lsp_server_real_supports_language (FoundryLspServer *self,
                                           const char       *language_id)
{
  g_auto(GStrv) languages = NULL;

  g_assert (FOUNDRY_IS_LSP_SERVER (self));
  g_assert (language_id != NULL);

  if (!(languages = foundry_lsp_server_dup_languages (self)))
    return FALSE;

  return g_strv_contains ((const char * const *)languages, language_id);
}

static void
foundry_lsp_server_get_property (GObject    *object,
                                 guint       prop_id,
                                 GValue     *value,
                                 GParamSpec *pspec)
{
  FoundryLspServer *self = FOUNDRY_LSP_SERVER (object);

  switch (prop_id)
    {
    case PROP_NAME:
      g_value_take_string (value, foundry_lsp_server_dup_name (self));
      break;

    case PROP_LANGUAGES:
      g_value_take_boxed (value, foundry_lsp_server_dup_languages (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_lsp_server_class_init (FoundryLspServerClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->get_property = foundry_lsp_server_get_property;

  klass->supports_language = foundry_lsp_server_real_supports_language;

  properties[PROP_NAME] =
    g_param_spec_string ("name", NULL, NULL,
                         NULL,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_LANGUAGES] =
    g_param_spec_boxed ("languages", NULL, NULL,
                        G_TYPE_STRV,
                        (G_PARAM_READABLE |
                         G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_lsp_server_init (FoundryLspServer *self)
{
}

char *
foundry_lsp_server_dup_name (FoundryLspServer *self)
{
  g_return_val_if_fail (FOUNDRY_IS_LSP_SERVER (self), NULL);

  if (FOUNDRY_LSP_SERVER_GET_CLASS (self)->dup_name)
    return FOUNDRY_LSP_SERVER_GET_CLASS (self)->dup_name (self);

  return g_strdup (G_OBJECT_TYPE_NAME (self));
}

/**
 * foundry_lsp_server_dup_languages:
 * @self: a [class@Foundry.LspServer]
 *
 * Gets a string array of languages supported by the LSP.
 *
 * Returns: (transfer full): a string array of languages
 */
char **
foundry_lsp_server_dup_languages (FoundryLspServer *self)
{
  g_return_val_if_fail (FOUNDRY_IS_LSP_SERVER (self), NULL);

  if (FOUNDRY_LSP_SERVER_GET_CLASS (self)->dup_languages)
    return FOUNDRY_LSP_SERVER_GET_CLASS (self)->dup_languages (self);

  return g_new0 (char *, 1);
}

/**
 * foundry_lsp_server_prepare:
 * @self: a #FoundryLspServer
 * @pipeline: (nullable): a [class@Foundry.BuildPipeline]
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to a
 *   [class@Foundry.LspClient] or rejects with error
 */
DexFuture *
foundry_lsp_server_prepare (FoundryLspServer       *self,
                            FoundryBuildPipeline   *pipeline,
                            FoundryProcessLauncher *launcher)
{
  dex_return_error_if_fail (FOUNDRY_IS_LSP_SERVER (self));
  dex_return_error_if_fail (!pipeline || FOUNDRY_IS_BUILD_PIPELINE (pipeline));
  dex_return_error_if_fail (FOUNDRY_IS_PROCESS_LAUNCHER (launcher));

  return FOUNDRY_LSP_SERVER_GET_CLASS (self)->prepare (self, pipeline, launcher);
}

gboolean
foundry_lsp_server_supports_language (FoundryLspServer *self,
                                      const char       *language_id)
{
  g_return_val_if_fail (FOUNDRY_IS_LSP_SERVER (self), FALSE);
  g_return_val_if_fail (language_id != NULL, FALSE);

  return FOUNDRY_LSP_SERVER_GET_CLASS (self)->supports_language (self, language_id);
}
