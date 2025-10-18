/* foundry-diagnostic-fix.c
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

#include "foundry-diagnostic-fix-private.h"
#include "foundry-text-edit.h"

enum {
  PROP_0,
  PROP_DESCRIPTION,
  PROP_TEXT_EDITS,
  N_PROPS
};

G_DEFINE_FINAL_TYPE (FoundryDiagnosticFix, foundry_diagnostic_fix, G_TYPE_OBJECT)

static GParamSpec *properties[N_PROPS];

static void
foundry_diagnostic_fix_finalize (GObject *object)
{
  FoundryDiagnosticFix *self = (FoundryDiagnosticFix *)object;

  g_clear_pointer (&self->description, g_free);
  g_clear_object (&self->text_edits);

  G_OBJECT_CLASS (foundry_diagnostic_fix_parent_class)->finalize (object);
}

static void
foundry_diagnostic_fix_get_property (GObject    *object,
                                     guint       prop_id,
                                     GValue     *value,
                                     GParamSpec *pspec)
{
  FoundryDiagnosticFix *self = FOUNDRY_DIAGNOSTIC_FIX (object);

  switch (prop_id)
    {
    case PROP_DESCRIPTION:
      self->description = g_value_dup_string (value);
      break;

    case PROP_TEXT_EDITS:
      self->text_edits = g_value_dup_object (value);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_diagnostic_fix_class_init (FoundryDiagnosticFixClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->finalize = foundry_diagnostic_fix_finalize;
  object_class->get_property = foundry_diagnostic_fix_get_property;

  properties[PROP_DESCRIPTION] =
    g_param_spec_string ("description", NULL, NULL,
                         NULL,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_TEXT_EDITS] =
    g_param_spec_object ("text-edits", NULL, NULL,
                         G_TYPE_LIST_MODEL,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_diagnostic_fix_init (FoundryDiagnosticFix *self)
{
}

/**
 * foundry_diagnostic_fix_dup_description:
 * @self: a [class@Foundry.DiagnosticFix]
 *
 * Since: 1.1
 */
char *
foundry_diagnostic_fix_dup_description (FoundryDiagnosticFix *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DIAGNOSTIC_FIX (self), NULL);

  return g_strdup (self->description);
}

/**
 * foundry_diagnostic_fix_list_text_edits:
 * @self: a [class@Foundry.DiagnosticFix]
 *
 * Gets the list of changes to be applied to fix the diagnostic.
 *
 * Returns: (transfer full): a [iface@Gio.ListModel] of
 *   [class@Foundry.TextEdit].
 *
 * Since: 1.1
 */
GListModel *
foundry_diagnostic_fix_list_text_edits (FoundryDiagnosticFix *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DIAGNOSTIC_FIX (self), NULL);
  g_return_val_if_fail (G_IS_LIST_MODEL (self->text_edits), NULL);

  return g_object_ref (self->text_edits);
}
