/* foundry-diagnostic.c
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

#include "foundry-diagnostic-private.h"
#include "foundry-diagnostic-range.h"

enum {
  PROP_0,
  PROP_FILE,
  PROP_HAS_FIX,
  PROP_LINE,
  PROP_LINE_OFFSET,
  PROP_MESSAGE,
  PROP_MARKUP,
  PROP_PATH,
  PROP_RANGES,
  PROP_RULE_ID,
  PROP_SEVERITY,
  N_PROPS
};

G_DEFINE_FINAL_TYPE (FoundryDiagnostic, foundry_diagnostic, G_TYPE_OBJECT)

static GParamSpec *properties[N_PROPS];

static void
foundry_diagnostic_finalize (GObject *object)
{
  FoundryDiagnostic *self = (FoundryDiagnostic *)object;

  g_clear_pointer (&self->message, g_free);
  g_clear_pointer (&self->rule_id, g_free);
  g_clear_object (&self->ranges);
  g_clear_object (&self->file);
  g_clear_object (&self->markup);
  g_clear_object (&self->fixes);

  G_OBJECT_CLASS (foundry_diagnostic_parent_class)->finalize (object);
}

static void
foundry_diagnostic_get_property (GObject    *object,
                                 guint       prop_id,
                                 GValue     *value,
                                 GParamSpec *pspec)
{
  FoundryDiagnostic *self = FOUNDRY_DIAGNOSTIC (object);

  switch (prop_id)
    {
    case PROP_FILE:
      g_value_take_object (value, foundry_diagnostic_dup_file (self));
      break;

    case PROP_HAS_FIX:
      g_value_set_boolean (value, foundry_diagnostic_has_fix (self));
      break;

    case PROP_LINE:
      g_value_set_uint (value, foundry_diagnostic_get_line (self));
      break;

    case PROP_LINE_OFFSET:
      g_value_set_uint (value, foundry_diagnostic_get_line_offset (self));
      break;

    case PROP_MARKUP:
      g_value_take_object (value, foundry_diagnostic_dup_markup (self));
      break;

    case PROP_MESSAGE:
      g_value_take_string (value, foundry_diagnostic_dup_message (self));
      break;

    case PROP_PATH:
      g_value_take_string (value, foundry_diagnostic_dup_path (self));
      break;

    case PROP_RANGES:
      g_value_take_object (value, foundry_diagnostic_list_ranges (self));
      break;

    case PROP_RULE_ID:
      g_value_take_string (value, foundry_diagnostic_dup_rule_id (self));
      break;

    case PROP_SEVERITY:
      g_value_set_enum (value, foundry_diagnostic_get_severity (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_diagnostic_class_init (FoundryDiagnosticClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->finalize = foundry_diagnostic_finalize;
  object_class->get_property = foundry_diagnostic_get_property;

  properties[PROP_FILE] =
    g_param_spec_object ("file", NULL, NULL,
                         G_TYPE_FILE,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_HAS_FIX] =
    g_param_spec_boolean ("has-fix", NULL, NULL,
                          FALSE,
                          (G_PARAM_READABLE |
                           G_PARAM_STATIC_STRINGS));

  properties[PROP_LINE] =
    g_param_spec_uint ("line", NULL, NULL,
                       0, G_MAXUINT, 0,
                       (G_PARAM_READABLE |
                        G_PARAM_STATIC_STRINGS));

  properties[PROP_LINE_OFFSET] =
    g_param_spec_uint ("line-offset", NULL, NULL,
                       0, G_MAXUINT, 0,
                       (G_PARAM_READABLE |
                        G_PARAM_STATIC_STRINGS));

  properties[PROP_MARKUP] =
    g_param_spec_object ("markup", NULL, NULL,
                         FOUNDRY_TYPE_MARKUP,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_MESSAGE] =
    g_param_spec_string ("message", NULL, NULL,
                         NULL,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_PATH] =
    g_param_spec_string ("path", NULL, NULL,
                         NULL,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_RANGES] =
    g_param_spec_object ("ranges", NULL, NULL,
                         G_TYPE_LIST_MODEL,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  /**
   * FoundryDiagnostic:rule-id:
   *
   * The "rule-id" property is an identifier to the type of rule for the
   * diagnostic.
   *
   * Not all diagnostics may have this, but it is generally useful to provide
   * from diagnostic providers where a known rule is applied. For example, many
   * compilers have a rule identifier which may be silenced by matching the
   * value of this property.
   *
   * Since: 1.1
   */
  properties[PROP_RULE_ID] =
    g_param_spec_string ("rule-id", NULL, NULL,
                         NULL,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_SEVERITY] =
    g_param_spec_enum ("severity", NULL, NULL,
                       FOUNDRY_TYPE_DIAGNOSTIC_SEVERITY,
                       FOUNDRY_DIAGNOSTIC_NOTE,
                       (G_PARAM_READABLE |
                        G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_diagnostic_init (FoundryDiagnostic *self)
{
}

/**
 * foundry_diagnostic_get_line:
 * @self: a [class@Foundry.Diagnostic]
 *
 * Gets the line number of the diagnostic.
 *
 * A value of 0 indicates that the line is unset.
 *
 * Returns: A line number starting from 1, or 0 to indicate unset.
 */
guint
foundry_diagnostic_get_line (FoundryDiagnostic *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DIAGNOSTIC (self), 0);

  return self->line;
}

/**
 * foundry_diagnostic_get_line_offset:
 * @self: a [class@Foundry.Diagnostic]
 *
 * Gets the offset within the line of the diagnostic.
 *
 * A value of 0 indicates that the line offset is unset.
 *
 * Returns: A character offset within the line starting from 1,
 *  or 0 to indicate unset.
 */
guint
foundry_diagnostic_get_line_offset (FoundryDiagnostic *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DIAGNOSTIC (self), 0);

  return self->line_offset;
}

/**
 * foundry_diagnostic_dup_message:
 * @self: a #FoundryDiagnostic
 *
 * Gets the message for the diagnostic, if any.
 *
 * Returns: (transfer full) (nullable): a message string or %NULL
 */
char *
foundry_diagnostic_dup_message (FoundryDiagnostic *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DIAGNOSTIC (self), NULL);

  if (self->message != NULL)
    return g_strdup (self->message);

  if (self->markup != NULL)
    {
      g_autoptr(GBytes) contents = foundry_markup_dup_contents (self->markup);

      if (contents != NULL)
        return g_strndup (g_bytes_get_data (contents, NULL),
                          g_bytes_get_size (contents));
    }

  return NULL;
}

/**
 * foundry_diagnostic_dup_path:
 * @self: a #FoundryDiagnostic
 *
 * Gets the path for the diagnostic, if any.
 *
 * Returns: (transfer full) (nullable): a string or %NULL
 */
char *
foundry_diagnostic_dup_path (FoundryDiagnostic *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DIAGNOSTIC (self), NULL);

  if (self->file == NULL || !g_file_is_native (self->file))
    return NULL;

  return g_file_get_path (self->file);
}

FoundryDiagnosticSeverity
foundry_diagnostic_get_severity (FoundryDiagnostic *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DIAGNOSTIC (self), 0);

  return self->severity;
}

/**
 * foundry_diagnostic_list_ranges:
 * @self: a #FoundryDiagnostic
 *
 * Gets the available ranges as a #GListModel of #FoundryDiagnosticRange.
 *
 * Returns: (transfer full): a #GListModel or %NULL if there are
 *   no ranges associated with this diagnostic.
 */
GListModel *
foundry_diagnostic_list_ranges (FoundryDiagnostic *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DIAGNOSTIC (self), NULL);

  if (self->ranges != NULL)
    return g_object_ref (self->ranges);

  return G_LIST_MODEL (g_list_store_new (FOUNDRY_TYPE_DIAGNOSTIC_RANGE));
}

/**
 * foundry_diagnostic_dup_file:
 * @self: a #FoundryDiagnostic
 *
 * Returns: (transfer full) (nullable): a #GFile or %NULL
 */
GFile *
foundry_diagnostic_dup_file (FoundryDiagnostic *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DIAGNOSTIC (self), NULL);

  return self->file ? g_object_ref (self->file) : NULL;
}

/**
 * foundry_diagnostic_dup_markup:
 * @self: a [class@Foundry.Diagnostic]
 *
 * Get the markup for the diagnostic, if any.
 *
 * Returns: (transfer full) (nullable):
 */
FoundryMarkup *
foundry_diagnostic_dup_markup (FoundryDiagnostic *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DIAGNOSTIC (self), NULL);

  return self->markup ? g_object_ref (self->markup) : NULL;
}

/**
 * foundry_diagnostic_dup_rule_id:
 * @self: a [class@Foundry.Diagnostic]
 *
 * Gets the [property@Foundry.Diagnostic:rule-id] property.
 *
 * Returns: (transfer full) (nullable):
 *
 * Since: 1.1
 */
char *
foundry_diagnostic_dup_rule_id (FoundryDiagnostic *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DIAGNOSTIC (self), NULL);

  return g_strdup (self->rule_id);
}

/**
 * foundry_diagnostic_list_fixes:
 * @self: a [class@Foundry.Diagnostic]
 *
 * Gets the available [class@Foundry.DiagnosticFix] for the diagnostic.
 *
 * Returns: (transfer full) (nullable): a [iface@Gio.ListModel] of
 *   [class@Foundry.DiagnosticFix]
 *
 * Since: 1.1
 */
GListModel *
foundry_diagnostic_list_fixes (FoundryDiagnostic *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DIAGNOSTIC (self), NULL);

  if (self->fixes == NULL)
    return NULL;

  return g_object_ref (G_LIST_MODEL (self->fixes));
}

/**
 * foundry_diagnostic_has_fix:
 * @self: a [class@Foundry.Diagnostic]
 *
 * Since: 1.1
 */
gboolean
foundry_diagnostic_has_fix (FoundryDiagnostic *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DIAGNOSTIC (self), FALSE);

  return self->fixes != NULL && g_list_model_get_n_items (self->fixes) > 0;
}

G_DEFINE_ENUM_TYPE (FoundryDiagnosticSeverity,
                    foundry_diagnostic_severity,
                    G_DEFINE_ENUM_VALUE (FOUNDRY_DIAGNOSTIC_IGNORED, "ignored"),
                    G_DEFINE_ENUM_VALUE (FOUNDRY_DIAGNOSTIC_NOTE, "note"),
                    G_DEFINE_ENUM_VALUE (FOUNDRY_DIAGNOSTIC_UNUSED, "unused"),
                    G_DEFINE_ENUM_VALUE (FOUNDRY_DIAGNOSTIC_DEPRECATED, "deprecated"),
                    G_DEFINE_ENUM_VALUE (FOUNDRY_DIAGNOSTIC_WARNING, "warning"),
                    G_DEFINE_ENUM_VALUE (FOUNDRY_DIAGNOSTIC_ERROR, "error"),
                    G_DEFINE_ENUM_VALUE (FOUNDRY_DIAGNOSTIC_FATAL, "fatal"))
