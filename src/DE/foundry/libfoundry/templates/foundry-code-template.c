/* foundry-code-template.c
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

#include "foundry-code-template.h"
#include "foundry-context.h"

typedef struct
{
  GWeakRef context_wr;
} FoundryCodeTemplatePrivate;

enum {
  PROP_0,
  PROP_CONTEXT,
  N_PROPS
};

G_DEFINE_ABSTRACT_TYPE_WITH_PRIVATE (FoundryCodeTemplate, foundry_code_template, FOUNDRY_TYPE_TEMPLATE)

static GParamSpec *properties[N_PROPS];

static void
foundry_code_template_finalize (GObject *object)
{
  FoundryCodeTemplate *self = (FoundryCodeTemplate *)object;
  FoundryCodeTemplatePrivate *priv = foundry_code_template_get_instance_private (self);

  g_weak_ref_clear (&priv->context_wr);

  G_OBJECT_CLASS (foundry_code_template_parent_class)->finalize (object);
}

static void
foundry_code_template_get_property (GObject    *object,
                                    guint       prop_id,
                                    GValue     *value,
                                    GParamSpec *pspec)
{
  FoundryCodeTemplate *self = FOUNDRY_CODE_TEMPLATE (object);

  switch (prop_id)
    {
    case PROP_CONTEXT:
      g_value_take_object (value, foundry_code_template_dup_context (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_code_template_set_property (GObject      *object,
                                    guint         prop_id,
                                    const GValue *value,
                                    GParamSpec   *pspec)
{
  FoundryCodeTemplate *self = FOUNDRY_CODE_TEMPLATE (object);
  FoundryCodeTemplatePrivate *priv = foundry_code_template_get_instance_private (self);

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
foundry_code_template_class_init (FoundryCodeTemplateClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->finalize = foundry_code_template_finalize;
  object_class->get_property = foundry_code_template_get_property;
  object_class->set_property = foundry_code_template_set_property;

  properties[PROP_CONTEXT] =
    g_param_spec_object ("context", NULL, NULL,
                         FOUNDRY_TYPE_CONTEXT,
                         (G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_code_template_init (FoundryCodeTemplate *self)
{
  FoundryCodeTemplatePrivate *priv = foundry_code_template_get_instance_private (self);

  g_weak_ref_init (&priv->context_wr, NULL);
}

/**
 * foundry_code_template_dup_context:
 * @self: a [class@Foundry.CodeTemplate]
 *
 * Returns: (transfer full) (nullable):
 */
FoundryContext *
foundry_code_template_dup_context (FoundryCodeTemplate *self)
{
  FoundryCodeTemplatePrivate *priv = foundry_code_template_get_instance_private (self);

  g_return_val_if_fail (FOUNDRY_IS_CODE_TEMPLATE (self), NULL);

  return g_weak_ref_get (&priv->context_wr);
}
