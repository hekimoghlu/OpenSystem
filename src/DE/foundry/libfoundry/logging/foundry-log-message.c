/* foundry-log-message.c
 *
 * Copyright 2023-2024 Christian Hergert <chergert@redhat.com>
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

#include "foundry-log-message-private.h"

enum {
  PROP_0,
  PROP_CREATED_AT,
  PROP_DOMAIN,
  PROP_MESSAGE,
  PROP_SEVERITY,
  N_PROPS
};

G_DEFINE_FINAL_TYPE (FoundryLogMessage, foundry_log_message, G_TYPE_OBJECT)

static GParamSpec *properties [N_PROPS];

static void
foundry_log_message_finalize (GObject *object)
{
  FoundryLogMessage *self = (FoundryLogMessage *)object;

  g_clear_pointer (&self->message, g_free);
  g_clear_pointer (&self->created_at, g_date_time_unref);

  G_OBJECT_CLASS (foundry_log_message_parent_class)->finalize (object);
}

static void
foundry_log_message_get_property (GObject    *object,
                                  guint       prop_id,
                                  GValue     *value,
                                  GParamSpec *pspec)
{
  FoundryLogMessage *self = FOUNDRY_LOG_MESSAGE (object);

  switch (prop_id)
    {
    case PROP_DOMAIN:
      g_value_set_static_string (value, foundry_log_message_get_domain (self));
      break;

    case PROP_CREATED_AT:
      g_value_take_boxed (value, foundry_log_message_dup_created_at (self));
      break;

    case PROP_MESSAGE:
      g_value_take_string (value, foundry_log_message_dup_message (self));
      break;

    case PROP_SEVERITY:
      g_value_set_uint (value, foundry_log_message_get_severity (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_log_message_set_property (GObject      *object,
                                  guint         prop_id,
                                  const GValue *value,
                                  GParamSpec   *pspec)
{
  FoundryLogMessage *self = FOUNDRY_LOG_MESSAGE (object);

  switch (prop_id)
    {
    case PROP_DOMAIN:
      self->domain = g_intern_string (g_value_get_string (value));
      break;

    case PROP_MESSAGE:
      self->message = g_value_dup_string (value);
      break;

    case PROP_CREATED_AT:
      self->created_at = g_value_dup_boxed (value);
      break;

    case PROP_SEVERITY:
      self->severity = g_value_get_uint (value);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_log_message_class_init (FoundryLogMessageClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->finalize = foundry_log_message_finalize;
  object_class->get_property = foundry_log_message_get_property;
  object_class->set_property = foundry_log_message_set_property;

  properties [PROP_MESSAGE] =
    g_param_spec_string ("message", NULL, NULL,
                         NULL,
                         (G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS));

  properties [PROP_CREATED_AT] =
    g_param_spec_boxed ("created-at", NULL, NULL,
                        G_TYPE_DATE_TIME,
                        (G_PARAM_READWRITE |
                         G_PARAM_CONSTRUCT_ONLY |
                         G_PARAM_STATIC_STRINGS));

  properties [PROP_DOMAIN] =
    g_param_spec_string ("domain", NULL, NULL,
                         NULL,
                         (G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS));

  properties [PROP_SEVERITY] =
    g_param_spec_uint ("severity", NULL, NULL,
                       0, G_MAXUINT, 0,
                       (G_PARAM_READWRITE |
                        G_PARAM_CONSTRUCT_ONLY |
                        G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_log_message_init (FoundryLogMessage *self)
{
}

FoundryLogMessage *
_foundry_log_message_new (GLogLevelFlags  severity,
                          const char     *domain,
                          char           *message,
                          GDateTime      *created_at)
{
  FoundryLogMessage *self;

  g_return_val_if_fail (domain != NULL, NULL);
  g_return_val_if_fail (message != NULL, NULL);

  self = g_object_new (FOUNDRY_TYPE_LOG_MESSAGE, NULL);
  self->domain = g_intern_string (domain);
  self->severity = severity;
  self->message = g_steal_pointer (&message);

  if (created_at == NULL)
    self->created_at = g_date_time_new_now_local ();
  else
    self->created_at = g_date_time_ref (created_at);

  return self;
}

/**
 * foundry_log_message_dup_message:
 * @self: a #FoundryLogMessage
 *
 * Gets the log message.
 *
 * Returns: (transfer full): A string containing the log message
 */
char *
foundry_log_message_dup_message (FoundryLogMessage *self)
{
  g_return_val_if_fail (FOUNDRY_IS_LOG_MESSAGE (self), NULL);

  return g_strdup (self->message);
}

/**
 * foundry_log_message_dup_created_at:
 * @self: a #FoundryLogMessage
 *
 * Gets the time the log item was created.
 *
 * Returns: (transfer full): a #GDateTime
 */
GDateTime *
foundry_log_message_dup_created_at (FoundryLogMessage *self)
{
  g_return_val_if_fail (FOUNDRY_IS_LOG_MESSAGE (self), NULL);

  return g_date_time_ref (self->created_at);
}

/**
 * foundry_log_message_get_domain:
 * @self: a #FoundryLogMessage
 *
 * Get the domain for the log item.
 */
const char *
foundry_log_message_get_domain (FoundryLogMessage *self)
{
  g_return_val_if_fail (FOUNDRY_IS_LOG_MESSAGE (self), NULL);

  return self->domain;
}

/**
 * foundry_log_message_get_severity
 * @self: a #FoundryLogMessage
 *
 * Gets the log item severity.
 */
GLogLevelFlags
foundry_log_message_get_severity (FoundryLogMessage *self)
{
  g_return_val_if_fail (FOUNDRY_IS_LOG_MESSAGE (self), 0);

  return self->severity;
}
