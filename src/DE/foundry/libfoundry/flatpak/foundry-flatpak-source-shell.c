/* foundry-flatpak-source-shell.c
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

#include "foundry-flatpak-source-shell.h"
#include "foundry-flatpak-source-private.h"
#include "foundry-util.h"

struct _FoundryFlatpakSourceShell
{
  FoundryFlatpakSource   parent_instance;
  char                 **commands;
};

enum {
  PROP_0,
  PROP_COMMANDS,
  N_PROPS
};

G_DEFINE_FINAL_TYPE (FoundryFlatpakSourceShell, foundry_flatpak_source_shell, FOUNDRY_TYPE_FLATPAK_SOURCE)

static void
foundry_flatpak_source_shell_finalize (GObject *object)
{
  FoundryFlatpakSourceShell *self = (FoundryFlatpakSourceShell *)object;

  g_clear_pointer (&self->commands, g_strfreev);

  G_OBJECT_CLASS (foundry_flatpak_source_shell_parent_class)->finalize (object);
}

static void
foundry_flatpak_source_shell_get_property (GObject    *object,
                                           guint       prop_id,
                                           GValue     *value,
                                           GParamSpec *pspec)
{
  FoundryFlatpakSourceShell *self = FOUNDRY_FLATPAK_SOURCE_SHELL (object);

  switch (prop_id)
    {
    case PROP_COMMANDS:
      g_value_set_boxed (value, self->commands);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_flatpak_source_shell_set_property (GObject      *object,
                                           guint         prop_id,
                                           const GValue *value,
                                           GParamSpec   *pspec)
{
  FoundryFlatpakSourceShell *self = FOUNDRY_FLATPAK_SOURCE_SHELL (object);

  switch (prop_id)
    {
    case PROP_COMMANDS:
      foundry_set_strv (&self->commands, g_value_get_boxed (value));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_flatpak_source_shell_class_init (FoundryFlatpakSourceShellClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundryFlatpakSourceClass *source_class = FOUNDRY_FLATPAK_SOURCE_CLASS (klass);

  object_class->finalize = foundry_flatpak_source_shell_finalize;
  object_class->get_property = foundry_flatpak_source_shell_get_property;
  object_class->set_property = foundry_flatpak_source_shell_set_property;

  source_class->type = "shell";

  g_object_class_install_property (object_class,
                                   PROP_COMMANDS,
                                   g_param_spec_boxed ("commands",
                                                       NULL,
                                                       NULL,
                                                       G_TYPE_STRV,
                                                       G_PARAM_STATIC_STRINGS | G_PARAM_READWRITE));
}

static void
foundry_flatpak_source_shell_init (FoundryFlatpakSourceShell *self)
{
}
