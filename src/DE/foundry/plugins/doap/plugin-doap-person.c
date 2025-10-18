/* plugin-doap-person.c
 *
 * Copyright 2015-2025 Christian Hergert <chergert@redhat.com>
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

#include <glib/gi18n-lib.h>

#include "plugin-doap-person.h"

struct _PluginDoapPerson
{
  GObject parent_instance;

  char *email;
  char *name;
};

G_DEFINE_FINAL_TYPE (PluginDoapPerson, plugin_doap_person, G_TYPE_OBJECT)

enum {
  PROP_0,
  PROP_EMAIL,
  PROP_NAME,
  N_PROPS
};

static GParamSpec *properties [N_PROPS];

PluginDoapPerson *
plugin_doap_person_new (void)
{
  return g_object_new (PLUGIN_TYPE_DOAP_PERSON, NULL);
}

const char *
plugin_doap_person_get_name (PluginDoapPerson *self)
{
  g_return_val_if_fail (PLUGIN_IS_DOAP_PERSON (self), NULL);

  return self->name;
}

void
plugin_doap_person_set_name (PluginDoapPerson *self,
                             const char       *name)
{
  g_return_if_fail (PLUGIN_IS_DOAP_PERSON (self));

  if (g_set_str (&self->name, name))
    g_object_notify_by_pspec (G_OBJECT (self), properties [PROP_NAME]);
}

const char *
plugin_doap_person_get_email (PluginDoapPerson *self)
{
  g_return_val_if_fail (PLUGIN_IS_DOAP_PERSON (self), NULL);

  return self->email;
}

void
plugin_doap_person_set_email (PluginDoapPerson *self,
                              const char      *email)
{
  g_return_if_fail (PLUGIN_IS_DOAP_PERSON (self));

  if (g_set_str (&self->email, email))
    g_object_notify_by_pspec (G_OBJECT (self), properties [PROP_EMAIL]);
}

static void
plugin_doap_person_finalize (GObject *object)
{
  PluginDoapPerson *self = (PluginDoapPerson *)object;

  g_clear_pointer (&self->email, g_free);
  g_clear_pointer (&self->name, g_free);

  G_OBJECT_CLASS (plugin_doap_person_parent_class)->finalize (object);
}

static void
plugin_doap_person_get_property (GObject    *object,
                                 guint       prop_id,
                                 GValue     *value,
                                 GParamSpec *pspec)
{
  PluginDoapPerson *self = PLUGIN_DOAP_PERSON (object);

  switch (prop_id)
    {
    case PROP_EMAIL:
      g_value_set_string (value, plugin_doap_person_get_email (self));
      break;

    case PROP_NAME:
      g_value_set_string (value, plugin_doap_person_get_name (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
plugin_doap_person_set_property (GObject      *object,
                                 guint         prop_id,
                                 const GValue *value,
                                 GParamSpec   *pspec)
{
  PluginDoapPerson *self = PLUGIN_DOAP_PERSON (object);

  switch (prop_id)
    {
    case PROP_EMAIL:
      plugin_doap_person_set_email (self, g_value_get_string (value));
      break;

    case PROP_NAME:
      plugin_doap_person_set_name (self, g_value_get_string (value));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
plugin_doap_person_class_init (PluginDoapPersonClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->finalize = plugin_doap_person_finalize;
  object_class->get_property = plugin_doap_person_get_property;
  object_class->set_property = plugin_doap_person_set_property;

  properties[PROP_EMAIL] =
    g_param_spec_string ("email", NULL, NULL,
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

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
plugin_doap_person_init (PluginDoapPerson *self)
{
}
