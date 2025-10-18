/* foundry-template.c
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

#include "foundry-input.h"
#include "foundry-code-template.h"
#include "foundry-project-template.h"
#include "foundry-template.h"
#include "foundry-util.h"

G_DEFINE_ABSTRACT_TYPE (FoundryTemplate, foundry_template, G_TYPE_OBJECT)

enum {
  PROP_0,
  PROP_DESCRIPTION,
  PROP_ID,
  PROP_INPUT,
  PROP_TAGS,
  N_PROPS
};

static GParamSpec *properties[N_PROPS];

static void
foundry_template_get_property (GObject    *object,
                               guint       prop_id,
                               GValue     *value,
                               GParamSpec *pspec)
{
  FoundryTemplate *self = FOUNDRY_TEMPLATE (object);

  switch (prop_id)
    {
    case PROP_ID:
      g_value_take_string (value, foundry_template_dup_id (self));
      break;

    case PROP_DESCRIPTION:
      g_value_take_string (value, foundry_template_dup_description (self));
      break;

    case PROP_INPUT:
      g_value_take_object (value, foundry_template_dup_input (self));
      break;

    case PROP_TAGS:
      g_value_take_boxed (value, foundry_template_dup_tags (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_template_class_init (FoundryTemplateClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->get_property = foundry_template_get_property;

  properties[PROP_ID] =
    g_param_spec_string ("id", NULL, NULL,
                         NULL,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_DESCRIPTION] =
    g_param_spec_string ("description", NULL, NULL,
                         NULL,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_INPUT] =
    g_param_spec_object ("input", NULL, NULL,
                         FOUNDRY_TYPE_INPUT,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_TAGS] =
    g_param_spec_boxed ("tags", NULL, NULL,
                        G_TYPE_STRV,
                        (G_PARAM_READABLE |
                         G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_template_init (FoundryTemplate *self)
{
}

char *
foundry_template_dup_id (FoundryTemplate *self)
{
  g_return_val_if_fail (FOUNDRY_IS_TEMPLATE (self), NULL);

  if (FOUNDRY_TEMPLATE_GET_CLASS (self)->dup_id)
    return FOUNDRY_TEMPLATE_GET_CLASS (self)->dup_id (self);

  return NULL;
}

char *
foundry_template_dup_description (FoundryTemplate *self)
{
  g_return_val_if_fail (FOUNDRY_IS_TEMPLATE (self), NULL);

  if (FOUNDRY_TEMPLATE_GET_CLASS (self)->dup_description)
    return FOUNDRY_TEMPLATE_GET_CLASS (self)->dup_description (self);

  return NULL;
}

/**
 * foundry_template_dup_input:
 * @self: a [class@Foundry.Template]
 *
 * Returns: (transfer full) (nullable):
 */
FoundryInput *
foundry_template_dup_input (FoundryTemplate *self)
{
  g_return_val_if_fail (FOUNDRY_IS_TEMPLATE (self), NULL);

  if (FOUNDRY_TEMPLATE_GET_CLASS (self)->dup_input)
    return FOUNDRY_TEMPLATE_GET_CLASS (self)->dup_input (self);

  return NULL;
}

/**
 * foundry_template_expand:
 * @self: a [class@Foundry.Template]
 *
 * Expands the template based on the input parameters provided to the template.
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to a
 *   [iface@Gio.ListModel] of [class@Foundry.TemplateOutput].
 */
DexFuture *
foundry_template_expand (FoundryTemplate *self)
{
  dex_return_error_if_fail (FOUNDRY_IS_TEMPLATE (self));

  if (FOUNDRY_TEMPLATE_GET_CLASS (self)->expand)
    return FOUNDRY_TEMPLATE_GET_CLASS (self)->expand (self);

  return foundry_future_new_not_supported ();
}

static char **
strv_append (char       **strv,
             const char  *word)
{
  gsize len = strv ? g_strv_length (strv) : 0;

  if (strv != NULL)
    strv = g_realloc_n (strv, len + 2, sizeof *strv);
  else
    strv = g_new (char *, len + 2);

  strv[len++] = g_strdup (word);
  strv[len] = NULL;

  return strv;
}

/**
 * foundry_template_dup_tags:
 * @self: a [class@Foundry.Template]
 *
 * Gets tags describing the template such as "meson" or "flatpak".
 *
 * Returns: (transfer full) (nullable):
 */
char **
foundry_template_dup_tags (FoundryTemplate *self)
{
  g_auto(GStrv) tags = NULL;

  g_return_val_if_fail (FOUNDRY_IS_TEMPLATE (self), NULL);

  if (FOUNDRY_TEMPLATE_GET_CLASS (self)->dup_tags)
    tags = FOUNDRY_TEMPLATE_GET_CLASS (self)->dup_tags (self);

  if (FOUNDRY_IS_PROJECT_TEMPLATE (self))
    tags = strv_append (tags, "project");
  else if (FOUNDRY_IS_CODE_TEMPLATE (self))
    tags = strv_append (tags, "code");

  return g_steal_pointer (&tags);
}
