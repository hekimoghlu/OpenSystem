/* foundry-diagnostic-range.c
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

#include "foundry-diagnostic-range.h"

struct _FoundryDiagnosticRange
{
  GObject parent_instance;
  guint start_line;
  guint start_line_offset;
  guint end_line;
  guint end_line_offset;
};

enum {
  PROP_0,
  PROP_START_LINE,
  PROP_START_LINE_OFFSET,
  PROP_END_LINE,
  PROP_END_LINE_OFFSET,
  N_PROPS
};

G_DEFINE_FINAL_TYPE (FoundryDiagnosticRange, foundry_diagnostic_range, G_TYPE_OBJECT)

static GParamSpec *properties[N_PROPS];

static void
foundry_diagnostic_range_get_property (GObject    *object,
                                       guint       prop_id,
                                       GValue     *value,
                                       GParamSpec *pspec)
{
  FoundryDiagnosticRange *self = FOUNDRY_DIAGNOSTIC_RANGE (object);

  switch (prop_id)
    {
    case PROP_START_LINE:
      g_value_set_uint (value, self->start_line);
      break;

    case PROP_START_LINE_OFFSET:
      g_value_set_uint (value, self->start_line_offset);
      break;

    case PROP_END_LINE:
      g_value_set_uint (value, self->end_line);
      break;

    case PROP_END_LINE_OFFSET:
      g_value_set_uint (value, self->end_line_offset);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_diagnostic_range_set_property (GObject      *object,
                                       guint         prop_id,
                                       const GValue *value,
                                       GParamSpec   *pspec)
{
  FoundryDiagnosticRange *self = FOUNDRY_DIAGNOSTIC_RANGE (object);

  switch (prop_id)
    {
    case PROP_START_LINE:
      self->start_line = g_value_get_uint (value);
      break;

    case PROP_START_LINE_OFFSET:
      self->start_line_offset = g_value_get_uint (value);
      break;

    case PROP_END_LINE:
      self->end_line = g_value_get_uint (value);
      break;

    case PROP_END_LINE_OFFSET:
      self->end_line_offset = g_value_get_uint (value);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_diagnostic_range_class_init (FoundryDiagnosticRangeClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->get_property = foundry_diagnostic_range_get_property;
  object_class->set_property = foundry_diagnostic_range_set_property;

  properties[PROP_START_LINE] =
    g_param_spec_uint ("start-line", NULL, NULL,
                       0, G_MAXUINT, 0,
                       (G_PARAM_READWRITE |
                        G_PARAM_CONSTRUCT_ONLY |
                        G_PARAM_STATIC_STRINGS));

  properties[PROP_START_LINE_OFFSET] =
    g_param_spec_uint ("start-line-offset", NULL, NULL,
                       0, G_MAXUINT, 0,
                       (G_PARAM_READWRITE |
                        G_PARAM_CONSTRUCT_ONLY |
                        G_PARAM_STATIC_STRINGS));

  properties[PROP_END_LINE] =
    g_param_spec_uint ("end-line", NULL, NULL,
                       0, G_MAXUINT, 0,
                       (G_PARAM_READWRITE |
                        G_PARAM_CONSTRUCT_ONLY |
                        G_PARAM_STATIC_STRINGS));

  properties[PROP_END_LINE_OFFSET] =
    g_param_spec_uint ("end-line-offset", NULL, NULL,
                       0, G_MAXUINT, 0,
                       (G_PARAM_READWRITE |
                        G_PARAM_CONSTRUCT_ONLY |
                        G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_diagnostic_range_init (FoundryDiagnosticRange *self)
{
}

guint
foundry_diagnostic_range_get_start_line (FoundryDiagnosticRange *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DIAGNOSTIC_RANGE (self), 0);

  return self->start_line;
}

guint
foundry_diagnostic_range_get_start_line_offset (FoundryDiagnosticRange *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DIAGNOSTIC_RANGE (self), 0);

  return self->start_line_offset;
}

guint
foundry_diagnostic_range_get_end_line (FoundryDiagnosticRange *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DIAGNOSTIC_RANGE (self), 0);

  return self->end_line;
}

guint
foundry_diagnostic_range_get_end_line_offset (FoundryDiagnosticRange *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DIAGNOSTIC_RANGE (self), 0);

  return self->end_line_offset;
}
