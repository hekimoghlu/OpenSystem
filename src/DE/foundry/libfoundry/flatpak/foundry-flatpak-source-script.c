/* foundry-flatpak-source-script.c
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

#include "foundry-flatpak-source-script.h"
#include "foundry-flatpak-source-private.h"
#include "foundry-util.h"

struct _FoundryFlatpakSourceScript
{
  FoundryFlatpakSource   parent_instance;
  char                 **commands;
  char                  *dest_filename;
};

enum {
  PROP_0,
  PROP_COMMANDS,
  PROP_DEST_FILENAME,
  N_PROPS
};

G_DEFINE_FINAL_TYPE (FoundryFlatpakSourceScript, foundry_flatpak_source_script, FOUNDRY_TYPE_FLATPAK_SOURCE)

static void
foundry_flatpak_source_script_finalize (GObject *object)
{
  FoundryFlatpakSourceScript *self = (FoundryFlatpakSourceScript *)object;

  g_clear_pointer (&self->commands, g_strfreev);
  g_clear_pointer (&self->dest_filename, g_free);

  G_OBJECT_CLASS (foundry_flatpak_source_script_parent_class)->finalize (object);
}

static void
foundry_flatpak_source_script_get_property (GObject    *object,
                                            guint       prop_id,
                                            GValue     *value,
                                            GParamSpec *pspec)
{
  FoundryFlatpakSourceScript *self = FOUNDRY_FLATPAK_SOURCE_SCRIPT (object);

  switch (prop_id)
    {
      case PROP_DEST_FILENAME:
      g_value_set_string (value, self->dest_filename);
      break;

    case PROP_COMMANDS:
      g_value_set_boxed (value, self->commands);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_flatpak_source_script_set_property (GObject      *object,
                                            guint         prop_id,
                                            const GValue *value,
                                            GParamSpec   *pspec)
{
  FoundryFlatpakSourceScript *self = FOUNDRY_FLATPAK_SOURCE_SCRIPT (object);

  switch (prop_id)
    {
    case PROP_DEST_FILENAME:
      g_set_str (&self->dest_filename, g_value_get_string (value));
      break;

    case PROP_COMMANDS:
      foundry_set_strv (&self->commands, g_value_get_boxed (value));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_flatpak_source_script_class_init (FoundryFlatpakSourceScriptClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundryFlatpakSourceClass *source_class = FOUNDRY_FLATPAK_SOURCE_CLASS (klass);

  object_class->finalize = foundry_flatpak_source_script_finalize;
  object_class->get_property = foundry_flatpak_source_script_get_property;
  object_class->set_property = foundry_flatpak_source_script_set_property;

  source_class->type = "script";

  g_object_class_install_property (object_class,
                                   PROP_DEST_FILENAME,
                                   g_param_spec_string ("dest-filename",
                                                        NULL,
                                                        NULL,
                                                        NULL,
                                                        G_PARAM_STATIC_STRINGS | G_PARAM_READWRITE));

  g_object_class_install_property (object_class,
                                   PROP_COMMANDS,
                                   g_param_spec_boxed ("commands",
                                                        NULL,
                                                        NULL,
                                                        G_TYPE_STRV,
                                                        G_PARAM_STATIC_STRINGS | G_PARAM_READWRITE));
}

static void
foundry_flatpak_source_script_init (FoundryFlatpakSourceScript *self)
{
}
