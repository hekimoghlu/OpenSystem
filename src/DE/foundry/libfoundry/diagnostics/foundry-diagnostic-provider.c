/* foundry-diagnostic-provider.c
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

#include "foundry-diagnostic-provider-private.h"
#include "foundry-util.h"

typedef struct
{
  PeasPluginInfo *plugin_info;
} FoundryDiagnosticProviderPrivate;

G_DEFINE_ABSTRACT_TYPE_WITH_PRIVATE (FoundryDiagnosticProvider, foundry_diagnostic_provider, FOUNDRY_TYPE_CONTEXTUAL)

enum {
  PROP_0,
  PROP_PLUGIN_INFO,
  N_PROPS
};

static GParamSpec *properties[N_PROPS];

static void
foundry_diagnostic_provider_finalize (GObject *object)
{
  FoundryDiagnosticProvider *self = (FoundryDiagnosticProvider *)object;
  FoundryDiagnosticProviderPrivate *priv = foundry_diagnostic_provider_get_instance_private (self);

  g_clear_object (&priv->plugin_info);

  G_OBJECT_CLASS (foundry_diagnostic_provider_parent_class)->finalize (object);
}

static void
foundry_diagnostic_provider_get_property (GObject    *object,
                                          guint       prop_id,
                                          GValue     *value,
                                          GParamSpec *pspec)
{
  FoundryDiagnosticProvider *self = FOUNDRY_DIAGNOSTIC_PROVIDER (object);
  FoundryDiagnosticProviderPrivate *priv = foundry_diagnostic_provider_get_instance_private (self);

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
foundry_diagnostic_provider_set_property (GObject      *object,
                                          guint         prop_id,
                                          const GValue *value,
                                          GParamSpec   *pspec)
{
  FoundryDiagnosticProvider *self = FOUNDRY_DIAGNOSTIC_PROVIDER (object);
  FoundryDiagnosticProviderPrivate *priv = foundry_diagnostic_provider_get_instance_private (self);

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
foundry_diagnostic_provider_class_init (FoundryDiagnosticProviderClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->finalize = foundry_diagnostic_provider_finalize;
  object_class->get_property = foundry_diagnostic_provider_get_property;
  object_class->set_property = foundry_diagnostic_provider_set_property;

  properties[PROP_PLUGIN_INFO] =
    g_param_spec_object ("plugin-info", NULL, NULL,
                         PEAS_TYPE_PLUGIN_INFO,
                         (G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_diagnostic_provider_init (FoundryDiagnosticProvider *self)
{
}

/**
 * foundry_diagnostic_provider_dup_plugin_info:
 * @self: a [class@Foundry.DiagnosticProvider]
 *
 * Returns: (transfer full) (nullable):
 */
PeasPluginInfo *
foundry_diagnostic_provider_dup_plugin_info (FoundryDiagnosticProvider *self)
{
  FoundryDiagnosticProviderPrivate *priv = foundry_diagnostic_provider_get_instance_private (self);

  g_return_val_if_fail (FOUNDRY_IS_DIAGNOSTIC_PROVIDER (self), NULL);

  return priv->plugin_info ? g_object_ref (priv->plugin_info) : NULL;
}

DexFuture *
foundry_diagnostic_provider_load (FoundryDiagnosticProvider *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DIAGNOSTIC_PROVIDER (self), NULL);

  if (FOUNDRY_DIAGNOSTIC_PROVIDER_GET_CLASS (self)->load)
    return FOUNDRY_DIAGNOSTIC_PROVIDER_GET_CLASS (self)->load (self);

  return dex_future_new_true ();
}

DexFuture *
foundry_diagnostic_provider_unload (FoundryDiagnosticProvider *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DIAGNOSTIC_PROVIDER (self), NULL);

  if (FOUNDRY_DIAGNOSTIC_PROVIDER_GET_CLASS (self)->unload)
    return FOUNDRY_DIAGNOSTIC_PROVIDER_GET_CLASS (self)->unload (self);

  return dex_future_new_true ();
}

/**
 * foundry_diagnostic_provider_dup_name:
 * @self: a #FoundryDiagnosticProvider
 *
 * Gets a name for the provider that is expected to be displayed to
 * users such as "Flatpak".
 *
 * Returns: (transfer full): the name of the provider
 */
char *
foundry_diagnostic_provider_dup_name (FoundryDiagnosticProvider *self)
{
  char *ret = NULL;

  g_return_val_if_fail (FOUNDRY_IS_DIAGNOSTIC_PROVIDER (self), NULL);

  if (FOUNDRY_DIAGNOSTIC_PROVIDER_GET_CLASS (self)->dup_name)
    ret = FOUNDRY_DIAGNOSTIC_PROVIDER_GET_CLASS (self)->dup_name (self);

  if (ret == NULL)
    ret = g_strdup (G_OBJECT_TYPE_NAME (self));

  return g_steal_pointer (&ret);
}

/**
 * foundry_diagnostic_provider_diagnose:
 * @self: a #FoundryDiagnosticProvider
 * @file: (nullable): the [iface@Gio.File] of the underlying file, if any
 * @contents: (nullable): the [struct@GLib.Bytes] of file contents, or %NULL
 * @language: (nullable): the language code such as "c"
 *
 * Processes @file to extract diagnostics.
 *
 * Returns: (transfer full): a #DexFuture that resolves to a #GListModel
 *   of #FoundryDiagnostic.
 */
DexFuture *
foundry_diagnostic_provider_diagnose (FoundryDiagnosticProvider *self,
                                      GFile                     *file,
                                      GBytes                    *contents,
                                      const char                *language)
{
  dex_return_error_if_fail (FOUNDRY_IS_DIAGNOSTIC_PROVIDER (self));
  dex_return_error_if_fail (!file || G_IS_FILE (file));

  if (file == NULL && contents == NULL)
    return dex_future_new_reject (G_IO_ERROR,
                                  G_IO_ERROR_INVAL,
                                  "File or contents must be provided");

  if (FOUNDRY_DIAGNOSTIC_PROVIDER_GET_CLASS (self)->diagnose)
    return FOUNDRY_DIAGNOSTIC_PROVIDER_GET_CLASS (self)->diagnose (self, file, contents, language);

  return foundry_future_new_not_supported ();
}

/**
 * foundry_diagnostic_provider_list_all:
 * @self: a [class@Foundry.DiagnosticProvider]
 *
 * Lists all diagnostics known to the provider for the project.
 *
 * This is useful for applications which want to show a project-wide list of
 * diagnostics. Providers are encouraged to have this information cached
 * rather than try to scan the whole project when requested.
 *
 * For example, diagnostics for a build may include results from the most
 * recent build request.
 *
 * It is encouraged that providers update the listmodel as new diagnostics
 * are made available.
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to a
 *   [iface@Gio.ListModel] of [class@Foundry.Diagnostic].
 *
 * Since: 1.1
 */
DexFuture *
foundry_diagnostic_provider_list_all (FoundryDiagnosticProvider *self)
{
  dex_return_error_if_fail (FOUNDRY_IS_DIAGNOSTIC_PROVIDER (self));

  if (FOUNDRY_DIAGNOSTIC_PROVIDER_GET_CLASS (self)->list_all)
    return FOUNDRY_DIAGNOSTIC_PROVIDER_GET_CLASS (self)->list_all (self);

  return foundry_future_new_not_supported ();
}
