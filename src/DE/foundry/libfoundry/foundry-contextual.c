/* foundry-contextual.c
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

#include <json-glib/json-glib.h>

#include "foundry-build-manager.h"
#include "foundry-context.h"
#include "foundry-contextual-private.h"
#include "foundry-inhibitor-private.h"

typedef struct
{
  GWeakRef context_wr;
} FoundryContextualPrivate;

enum {
  PROP_0,
  PROP_CONTEXT,
  N_PROPS
};

G_DEFINE_QUARK (foundry-contextual, foundry_contextual_error)

static GParamSpec **
foundry_contextual_list_properties (JsonSerializable *serializable,
                                    guint            *n_pspecs)
{
  GParamSpec **pspecs;
  guint pos = 0;

  g_assert (G_IS_OBJECT (serializable));
  g_assert (n_pspecs != NULL);

  pspecs = g_object_class_list_properties (G_OBJECT_GET_CLASS (serializable), n_pspecs);

  while (pos < *n_pspecs)
    {
      if (G_IS_PARAM_SPEC_OBJECT (pspecs[pos]))
        {
          (*n_pspecs)--;

          if (pos < *n_pspecs)
            pspecs[pos] = pspecs[*n_pspecs];
        }
      else
        {
          pos++;
        }
    }

  return pspecs;
}

static void
serializable_iface_init (JsonSerializableIface *iface)
{
  iface->list_properties = foundry_contextual_list_properties;
}

G_DEFINE_ABSTRACT_TYPE_WITH_CODE (FoundryContextual, foundry_contextual, G_TYPE_OBJECT,
                                  G_ADD_PRIVATE (FoundryContextual)
                                  G_IMPLEMENT_INTERFACE (JSON_TYPE_SERIALIZABLE, serializable_iface_init))

static GParamSpec *properties[N_PROPS];

static void
foundry_contextual_finalize (GObject *object)
{
  FoundryContextual *self = (FoundryContextual *)object;
  FoundryContextualPrivate *priv = foundry_contextual_get_instance_private (self);

  g_weak_ref_clear (&priv->context_wr);

  G_OBJECT_CLASS (foundry_contextual_parent_class)->finalize (object);
}

static void
foundry_contextual_get_property (GObject    *object,
                                 guint       prop_id,
                                 GValue     *value,
                                 GParamSpec *pspec)
{
  FoundryContextual *self = FOUNDRY_CONTEXTUAL (object);

  switch (prop_id)
    {
    case PROP_CONTEXT:
      g_value_take_object (value, foundry_contextual_dup_context (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_contextual_set_property (GObject      *object,
                                 guint         prop_id,
                                 const GValue *value,
                                 GParamSpec   *pspec)
{
  FoundryContextual *self = FOUNDRY_CONTEXTUAL (object);
  FoundryContextualPrivate *priv = foundry_contextual_get_instance_private (self);

  switch (prop_id)
    {
    case PROP_CONTEXT:
      g_weak_ref_set (&priv->context_wr, g_value_get_object (value));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_contextual_class_init (FoundryContextualClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->finalize = foundry_contextual_finalize;
  object_class->get_property = foundry_contextual_get_property;
  object_class->set_property = foundry_contextual_set_property;

  properties[PROP_CONTEXT] =
    g_param_spec_object ("context", NULL, NULL,
                         FOUNDRY_TYPE_CONTEXT,
                         (G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_contextual_init (FoundryContextual *self)
{
  FoundryContextualPrivate *priv = foundry_contextual_get_instance_private (self);

  g_weak_ref_init (&priv->context_wr, NULL);
}

/**
 * foundry_contextual_dup_context:
 * @self: a #FoundryContextual
 *
 * Gets the #FoundryContext that @self is a part of while safely increasing
 * the reference count of the resulting #FoundryContext by 1.
 *
 * Returns: (transfer full) (nullable): a #FoundryContext or %NULL
 */
FoundryContext *
foundry_contextual_dup_context (FoundryContextual *self)
{
  FoundryContextualPrivate *priv = foundry_contextual_get_instance_private (self);

  g_return_val_if_fail (FOUNDRY_IS_CONTEXTUAL (self), NULL);

  return g_weak_ref_get (&priv->context_wr);
}

void
_foundry_contextual_invalidate_pipeline (FoundryContextual *self)
{
  g_autoptr(FoundryContext) context = NULL;

  g_return_if_fail (FOUNDRY_IS_CONTEXTUAL (self));

  if ((context = foundry_contextual_dup_context (self)))
    {
      g_autoptr(FoundryBuildManager) build_manager = foundry_context_dup_build_manager (context);

      if (build_manager != NULL)
        foundry_build_manager_invalidate (build_manager);
    }
}

void
foundry_contextual_log (FoundryContextual *self,
                        const char        *domain,
                        GLogLevelFlags     severity,
                        const char        *format,
                        ...)
{
  g_autoptr(FoundryContext) context = NULL;
  g_autofree char *message = NULL;
  FoundryContextualClass *klass;
  va_list args;

  g_return_if_fail (FOUNDRY_IS_CONTEXTUAL (self));

  klass = FOUNDRY_CONTEXTUAL_GET_CLASS (self);

  if (klass->log_domain != NULL)
    domain = klass->log_domain;

  va_start (args, format);
  foundry_context_logv (context, domain, severity, format, args);
  va_end (args);
}

/**
 * foundry_contextual_inhibit:
 * @self: a [class@Foundry.Contextual]
 *
 * Creates a new [class@Foundry.Inhibitor] that will keep the
 * [class@Foundry.Context] alive and prevent shutdown until
 * [method@Foundry.Inhibitor.uninhibit] is called or the
 * [class@Foundry.Inhibitor] is finalized, whichever comes first.
 *
 * If the context is already in shutdown, then %NULL is returned and
 * @error is set.
 *
 * Returns: (transfer full): a [class@Foundry.Inhibitor] or %NULL and
 *   @error is set.
 */
FoundryInhibitor *
foundry_contextual_inhibit (FoundryContextual  *self,
                            GError            **error)
{
  g_autoptr(FoundryContext) context = NULL;

  g_return_val_if_fail (FOUNDRY_IS_CONTEXTUAL (self), NULL);

  context = foundry_contextual_dup_context (self);

  return foundry_inhibitor_new (context, error);
}
