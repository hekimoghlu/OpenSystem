/* foundry-llm-provider.c
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

#include "foundry-llm-model.h"
#include "foundry-llm-provider-private.h"
#include "foundry-util.h"

typedef struct
{
  PeasPluginInfo *plugin_info;
} FoundryLlmProviderPrivate;

G_DEFINE_ABSTRACT_TYPE_WITH_PRIVATE (FoundryLlmProvider, foundry_llm_provider, FOUNDRY_TYPE_CONTEXTUAL)

enum {
  PROP_0,
  PROP_PLUGIN_INFO,
  N_PROPS
};

static GParamSpec *properties[N_PROPS];

static DexFuture *
foundry_llm_provider_real_load (FoundryLlmProvider *self)
{
  return dex_future_new_true ();
}

static DexFuture *
foundry_llm_provider_real_unload (FoundryLlmProvider *self)
{
  return dex_future_new_true ();
}

static void
foundry_llm_provider_dispose (GObject *object)
{
  FoundryLlmProvider *self = (FoundryLlmProvider *)object;
  FoundryLlmProviderPrivate *priv = foundry_llm_provider_get_instance_private (self);

  g_clear_object (&priv->plugin_info);

  G_OBJECT_CLASS (foundry_llm_provider_parent_class)->dispose (object);
}

static void
foundry_llm_provider_get_property (GObject    *object,
                                   guint       prop_id,
                                   GValue     *value,
                                   GParamSpec *pspec)
{
  FoundryLlmProvider *self = FOUNDRY_LLM_PROVIDER (object);
  FoundryLlmProviderPrivate *priv = foundry_llm_provider_get_instance_private (self);

  switch (prop_id)
    {
    case PROP_PLUGIN_INFO:
      g_value_set_object (value, priv->plugin_info);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_llm_provider_set_property (GObject      *object,
                                   guint         prop_id,
                                   const GValue *value,
                                   GParamSpec   *pspec)
{
  FoundryLlmProvider *self = FOUNDRY_LLM_PROVIDER (object);
  FoundryLlmProviderPrivate *priv = foundry_llm_provider_get_instance_private (self);

  switch (prop_id)
    {
    case PROP_PLUGIN_INFO:
      priv->plugin_info = g_value_dup_object (value);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_llm_provider_class_init (FoundryLlmProviderClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->dispose = foundry_llm_provider_dispose;
  object_class->get_property = foundry_llm_provider_get_property;
  object_class->set_property = foundry_llm_provider_set_property;

  klass->load = foundry_llm_provider_real_load;
  klass->unload = foundry_llm_provider_real_unload;

  properties[PROP_PLUGIN_INFO] =
    g_param_spec_object ("plugin-info", NULL, NULL,
                         PEAS_TYPE_PLUGIN_INFO,
                         (G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_llm_provider_init (FoundryLlmProvider *self)
{
}

/**
 * foundry_llm_provider_dup_plugin_info:
 * @self: a [class@Foundry.LlmProvider]
 *
 * Returns: (transfer full) (nullable):
 */
PeasPluginInfo *
foundry_llm_provider_dup_plugin_info (FoundryLlmProvider *self)
{
  FoundryLlmProviderPrivate *priv = foundry_llm_provider_get_instance_private (self);

  g_return_val_if_fail (FOUNDRY_IS_LLM_PROVIDER (self), NULL);

  return priv->plugin_info ? g_object_ref (priv->plugin_info) : NULL;
}

/**
 * foundry_llm_provider_load:
 * @self: a #FoundryLlmProvider
 *
 * Returns: (transfer full): a #DexFuture
 */
DexFuture *
foundry_llm_provider_load (FoundryLlmProvider *self)
{
  g_return_val_if_fail (FOUNDRY_IS_LLM_PROVIDER (self), NULL);

  return FOUNDRY_LLM_PROVIDER_GET_CLASS (self)->load (self);
}

/**
 * foundry_llm_provider_unload:
 * @self: a #FoundryLlmProvider
 *
 * Returns: (transfer full): a #DexFuture
 */
DexFuture *
foundry_llm_provider_unload (FoundryLlmProvider *self)
{
  g_return_val_if_fail (FOUNDRY_IS_LLM_PROVIDER (self), NULL);

  return FOUNDRY_LLM_PROVIDER_GET_CLASS (self)->unload (self);
}

/**
 * foundry_llm_provider_dup_name:
 * @self: a #FoundryLlmProvider
 *
 * Gets a name for the provider that is expected to be displayed to
 * users such as "Ollama".
 *
 * Returns: (transfer full): the name of the provider
 */
char *
foundry_llm_provider_dup_name (FoundryLlmProvider *self)
{
  char *ret = NULL;

  g_return_val_if_fail (FOUNDRY_IS_LLM_PROVIDER (self), NULL);

  if (FOUNDRY_LLM_PROVIDER_GET_CLASS (self)->dup_name)
    ret = FOUNDRY_LLM_PROVIDER_GET_CLASS (self)->dup_name (self);

  if (ret == NULL)
    ret = g_strdup (G_OBJECT_TYPE_NAME (self));

  return g_steal_pointer (&ret);
}

/**
 * foundry_llm_provider_list_models:
 * @self: a [class@Foundry.LlmProvider]
 *
 * List the models available from the provider.
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to a
 *   [iface@Gio.ListModel] of [class@Foundry.LlmModel].
 */
DexFuture *
foundry_llm_provider_list_models (FoundryLlmProvider *self)
{
  dex_return_error_if_fail (FOUNDRY_IS_LLM_PROVIDER (self));

  if (FOUNDRY_LLM_PROVIDER_GET_CLASS (self)->list_models)
    return FOUNDRY_LLM_PROVIDER_GET_CLASS (self)->list_models (self);

  return foundry_future_new_not_supported ();
}

/**
 * foundry_llm_provider_list_tools:
 * @self: a [class@Foundry.LlmProvider]
 *
 * List the tools available from the provider.
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to a
 *   [iface@Gio.ListModel] of [class@Foundry.LlmTool].
 */
DexFuture *
foundry_llm_provider_list_tools (FoundryLlmProvider *self)
{
  dex_return_error_if_fail (FOUNDRY_IS_LLM_PROVIDER (self));

  if (FOUNDRY_LLM_PROVIDER_GET_CLASS (self)->list_tools)
    return FOUNDRY_LLM_PROVIDER_GET_CLASS (self)->list_tools (self);

  return foundry_future_new_not_supported ();
}
