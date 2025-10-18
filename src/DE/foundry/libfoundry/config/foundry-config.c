/* foundry-config.c
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

#include "foundry-build-pipeline.h"
#include "foundry-config-manager.h"
#include "foundry-config-private.h"
#include "foundry-config-provider.h"
#include "foundry-device.h"
#include "foundry-sdk.h"
#include "foundry-sdk-manager.h"

typedef struct _FoundryConfigPrivate
{
  GWeakRef provider_wr;
  char *id;
  char *name;
} FoundryConfigPrivate;

enum {
  PROP_0,
  PROP_ACTIVE,
  PROP_BUILD_SYSTEM,
  PROP_CAN_DEFAULT,
  PROP_CONFIG_OPTS,
  PROP_PRIORITY,
  PROP_ID,
  PROP_NAME,
  PROP_PROVIDER,
  N_PROPS
};

G_DEFINE_ABSTRACT_TYPE_WITH_PRIVATE (FoundryConfig, foundry_config, FOUNDRY_TYPE_CONTEXTUAL)

G_DEFINE_ENUM_TYPE (FoundryLocality, foundry_locality,
                    G_DEFINE_ENUM_VALUE (FOUNDRY_LOCALITY_RUN, "run"),
                    G_DEFINE_ENUM_VALUE (FOUNDRY_LOCALITY_BUILD, "build"),
                    G_DEFINE_ENUM_VALUE (FOUNDRY_LOCALITY_TOOL, "tool"))

static GParamSpec *properties[N_PROPS];

static char *
foundry_config_real_dup_builddir (FoundryConfig        *self,
                                  FoundryBuildPipeline *pipeline)
{
  g_autoptr(FoundryContext) context = NULL;

  g_assert (FOUNDRY_IS_CONFIG (self));
  g_assert (FOUNDRY_IS_BUILD_PIPELINE (pipeline));

  context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (self));

  return foundry_context_cache_filename (context, "build", NULL);
}

static DexFuture *
foundry_config_real_resolve_sdk (FoundryConfig *self,
                                 FoundryDevice *device)
{
  g_autoptr(FoundryContext) context = NULL;

  g_assert (FOUNDRY_IS_CONFIG (self));
  g_assert (FOUNDRY_IS_DEVICE (device));

  if ((context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (self))))
    {
      g_autoptr(FoundrySdkManager) sdk_manager = foundry_context_dup_sdk_manager (context);

      return foundry_sdk_manager_find_by_id (sdk_manager, "host");
    }

  return dex_future_new_reject (G_IO_ERROR,
                                G_IO_ERROR_NOT_FOUND,
                                "Not found");
}

static void
foundry_config_finalize (GObject *object)
{
  FoundryConfig *self = (FoundryConfig *)object;
  FoundryConfigPrivate *priv = foundry_config_get_instance_private (self);

  g_weak_ref_clear (&priv->provider_wr);

  g_clear_pointer (&priv->name, g_free);
  g_clear_pointer (&priv->id, g_free);

  G_OBJECT_CLASS (foundry_config_parent_class)->finalize (object);
}

static void
foundry_config_get_property (GObject    *object,
                             guint       prop_id,
                             GValue     *value,
                             GParamSpec *pspec)
{
  FoundryConfig *self = FOUNDRY_CONFIG (object);

  switch (prop_id)
    {
    case PROP_ACTIVE:
      g_value_set_boolean (value, foundry_config_get_active (self));
      break;

    case PROP_BUILD_SYSTEM:
      g_value_take_string (value, foundry_config_dup_build_system (self));
      break;

    case PROP_CAN_DEFAULT:
      g_value_set_boolean (value, foundry_config_can_default (self, NULL));
      break;

    case PROP_CONFIG_OPTS:
      g_value_take_boxed (value, foundry_config_dup_config_opts (self));
      break;

    case PROP_PRIORITY:
      {
        guint priority;
        foundry_config_can_default (self, &priority);
        g_value_set_uint (value, priority);
        break;
      }

    case PROP_ID:
      g_value_take_string (value, foundry_config_dup_id (self));
      break;

    case PROP_NAME:
      g_value_take_string (value, foundry_config_dup_name (self));
      break;

    case PROP_PROVIDER:
      g_value_take_object (value, _foundry_config_dup_provider (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_config_set_property (GObject      *object,
                             guint         prop_id,
                             const GValue *value,
                             GParamSpec   *pspec)
{
  FoundryConfig *self = FOUNDRY_CONFIG (object);

  switch (prop_id)
    {
    case PROP_ID:
      foundry_config_set_id (self, g_value_get_string (value));
      break;

    case PROP_NAME:
      foundry_config_set_name (self, g_value_get_string (value));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_config_class_init (FoundryConfigClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundryConfigClass *config_class = FOUNDRY_CONFIG_CLASS (klass);

  object_class->finalize = foundry_config_finalize;
  object_class->get_property = foundry_config_get_property;
  object_class->set_property = foundry_config_set_property;

  config_class->resolve_sdk = foundry_config_real_resolve_sdk;
  config_class->dup_builddir = foundry_config_real_dup_builddir;

  properties[PROP_ACTIVE] =
    g_param_spec_boolean ("active", NULL, NULL,
                          FALSE,
                          (G_PARAM_READABLE |
                           G_PARAM_STATIC_STRINGS));

  properties[PROP_BUILD_SYSTEM] =
    g_param_spec_string ("build-system", NULL, NULL,
                         NULL,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_CAN_DEFAULT] =
    g_param_spec_boolean ("can-default", NULL, NULL,
                          FALSE,
                          (G_PARAM_READABLE |
                           G_PARAM_STATIC_STRINGS));

  properties[PROP_CONFIG_OPTS] =
    g_param_spec_boxed ("config-opts", NULL, NULL,
                        G_TYPE_STRV,
                        (G_PARAM_READABLE |
                         G_PARAM_STATIC_STRINGS));

  properties[PROP_PRIORITY] =
    g_param_spec_uint ("priority", NULL, NULL,
                       0, G_MAXUINT, 0,
                       (G_PARAM_READABLE |
                        G_PARAM_STATIC_STRINGS));

  properties[PROP_ID] =
    g_param_spec_string ("id", NULL, NULL,
                         NULL,
                         (G_PARAM_READWRITE |
                          G_PARAM_EXPLICIT_NOTIFY |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_NAME] =
    g_param_spec_string ("name", NULL, NULL,
                         NULL,
                         (G_PARAM_READWRITE |
                          G_PARAM_EXPLICIT_NOTIFY |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_PROVIDER] =
    g_param_spec_object ("provider", NULL, NULL,
                         FOUNDRY_TYPE_CONFIG_PROVIDER,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_config_init (FoundryConfig *self)
{
  FoundryConfigPrivate *priv = foundry_config_get_instance_private (self);

  g_weak_ref_init (&priv->provider_wr, NULL);
}

/**
 * foundry_config_dup_name:
 * @self: a #FoundryConfig
 *
 * Gets the user-visible name for the configuration.
 *
 * Returns: (transfer full): a newly allocated string
 */
char *
foundry_config_dup_name (FoundryConfig *self)
{
  FoundryConfigPrivate *priv = foundry_config_get_instance_private (self);

  g_return_val_if_fail (FOUNDRY_IS_CONFIG (self), NULL);

  return g_strdup (priv->name);
}

/**
 * foundry_config_set_name:
 * @self: a #FoundryConfig
 *
 * Set the user-visible name of the config.
 *
 * This should only be called by implementations of #FoundryConfigProvider.
 */
void
foundry_config_set_name (FoundryConfig *self,
                         const char *name)
{
  FoundryConfigPrivate *priv = foundry_config_get_instance_private (self);

  g_return_if_fail (FOUNDRY_IS_CONFIG (self));

  if (g_set_str (&priv->name, name))
    g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_NAME]);
}

FoundryConfigProvider *
_foundry_config_dup_provider (FoundryConfig *self)
{
  FoundryConfigPrivate *priv = foundry_config_get_instance_private (self);

  g_return_val_if_fail (FOUNDRY_IS_CONFIG (self), NULL);

  return g_weak_ref_get (&priv->provider_wr);
}

void
_foundry_config_set_provider (FoundryConfig         *self,
                              FoundryConfigProvider *provider)
{
  FoundryConfigPrivate *priv = foundry_config_get_instance_private (self);

  g_return_if_fail (FOUNDRY_IS_CONFIG (self));
  g_return_if_fail (!provider || FOUNDRY_IS_CONFIG_PROVIDER (provider));

  g_weak_ref_set (&priv->provider_wr, provider);
}

gboolean
foundry_config_get_active (FoundryConfig *self)
{
  g_autoptr(FoundryContext) context = NULL;

  g_return_val_if_fail (FOUNDRY_IS_CONFIG (self), FALSE);

  if ((context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (self))))
    {
      g_autoptr(FoundryConfigManager) config_manager = foundry_context_dup_config_manager (context);
      g_autoptr(FoundryConfig) config = foundry_config_manager_dup_config (config_manager);

      return config == self;
    }

  return FALSE;
}

/**
 * foundry_config_dup_id:
 * @self: a #FoundryConfig
 *
 * Returns: (transfer full) (nullable): the identifier of the config
 */
char *
foundry_config_dup_id (FoundryConfig *self)
{
  FoundryConfigPrivate *priv = foundry_config_get_instance_private (self);

  g_return_val_if_fail (FOUNDRY_IS_CONFIG (self), NULL);

  return g_strdup (priv->id);
}

/**
 * foundry_config_set_id:
 * @self: a #FoundryConfig
 * @id: the unique identifier for the config
 *
 * Sets the identifier of the config.
 *
 * This should only be called by [class@Foundry.ConfigProvider] on their
 * [class@Foundry.Config] before they have been registered.
 */
void
foundry_config_set_id (FoundryConfig *self,
                       const char    *id)
{
  FoundryConfigPrivate *priv = foundry_config_get_instance_private (self);

  g_return_if_fail (FOUNDRY_IS_CONFIG (self));

  if (g_set_str (&priv->id, id))
    g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_ID]);
}

/**
 * foundry_config_resolve_sdk:
 * @self: a [class@Foundry.Config]
 * @device: a [class@Foundry.Device]
 *
 * Tries to locate the preferred SDK for a configuration and device.
 *
 * This might be used to locate an SDK which is not yet installed but would
 * need to be installed to properly setup a build pipeline.
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to a
 *   [class@Foundry.Sdk] or rejects with error.
 */
DexFuture *
foundry_config_resolve_sdk (FoundryConfig *self,
                            FoundryDevice *device)
{
  dex_return_error_if_fail (FOUNDRY_IS_CONFIG (self));

  return FOUNDRY_CONFIG_GET_CLASS (self)->resolve_sdk (self, device);
}

/**
 * foundry_config_dup_environ:
 *
 * Gets the environment variables to use for a particular locality.
 *
 * Returns: (transfer full) (nullable): an array of UTF-8 encoded strings
 *   or %NULL to use the default environment.
 */
char **
foundry_config_dup_environ (FoundryConfig   *self,
                            FoundryLocality  locality)
{
  g_return_val_if_fail (FOUNDRY_IS_CONFIG (self), NULL);
  g_return_val_if_fail (locality < FOUNDRY_LOCALITY_LAST, NULL);

  if (FOUNDRY_CONFIG_GET_CLASS (self)->dup_environ)
    return FOUNDRY_CONFIG_GET_CLASS (self)->dup_environ (self, locality);

  return NULL;
}

/**
 * foundry_config_can_default:
 * @self: a [class@Foundry.Config]
 * @priority: (out): the priority of the configuration
 *
 * Returns: %TRUE if @self can be the default configuration when loading
 *   a project for the first time.
 */
gboolean
foundry_config_can_default (FoundryConfig *self,
                            guint         *priority)
{
  guint unset;

  g_return_val_if_fail (FOUNDRY_IS_CONFIG (self), FALSE);

  if (priority == NULL)
    priority = &unset;

  *priority = 0;

  if (FOUNDRY_CONFIG_GET_CLASS (self)->can_default)
    return FOUNDRY_CONFIG_GET_CLASS (self)->can_default (self, priority);

  return FALSE;
}

/**
 * foundry_config_dup_build_system:
 * @self: a [class@Foundry.Config]
 *
 * The build system the configuration specifies to be used.
 *
 * Returns: (transfer full) (nullable): a build system or %NULL
 */
char *
foundry_config_dup_build_system (FoundryConfig *self)
{
  g_return_val_if_fail (FOUNDRY_IS_CONFIG (self), NULL);

  if (FOUNDRY_CONFIG_GET_CLASS (self)->dup_build_system)
    return FOUNDRY_CONFIG_GET_CLASS (self)->dup_build_system (self);

  return NULL;
}

/**
 * foundry_config_dup_builddir:
 * @self: a [class@Foundry.Config]
 * @pipeline: the [class@Foundry.BuildPipeline] which will perform the build
 *
 * Determines where the project build should occur.
 *
 * Returns: (transfer full): the directory where the build should occur
 */
char *
foundry_config_dup_builddir (FoundryConfig        *self,
                             FoundryBuildPipeline *pipeline)
{
  g_return_val_if_fail (FOUNDRY_IS_CONFIG (self), NULL);
  g_return_val_if_fail (FOUNDRY_IS_BUILD_PIPELINE (pipeline), NULL);

  return FOUNDRY_CONFIG_GET_CLASS (self)->dup_builddir (self, pipeline);
}

/**
 * foundry_config_dup_config_opts:
 * @self: a #FoundryConfig
 *
 * The config options.
 *
 * This is generally passed to something like `meson setup` or `cmake` when
 * configuring the project for a build.
 *
 * Returns: (transfer full) (nullable): the config options for
 *   configuring the project.
 */
char **
foundry_config_dup_config_opts (FoundryConfig *self)
{
  g_return_val_if_fail (FOUNDRY_IS_CONFIG (self), NULL);

  if (FOUNDRY_CONFIG_GET_CLASS (self)->dup_config_opts)
    return FOUNDRY_CONFIG_GET_CLASS (self)->dup_config_opts (self);

  return NULL;
}

/**
 * foundry_config_dup_default_command:
 * @self: a #FoundryConfig
 *
 * Gets the default command for the config, if any.
 *
 * Returns: (transfer full) (nullable): a [class@Foundry.Config] or %NULL.
 */
FoundryCommand *
foundry_config_dup_default_command (FoundryConfig *self)
{
  g_return_val_if_fail (FOUNDRY_IS_CONFIG (self), NULL);

  if (FOUNDRY_CONFIG_GET_CLASS (self)->dup_default_command)
    return FOUNDRY_CONFIG_GET_CLASS (self)->dup_default_command (self);

  return NULL;
}
