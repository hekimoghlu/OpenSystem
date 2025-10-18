/* foundry-signal-responder.c
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

#include "foundry-signal-responder-private.h"

struct _FoundrySignalResponder
{
  FoundryResponder   parent_instance;
  GWeakRef           object_wr;
  char              *detailed_signal;
  GtkExpression     *return_value;
  GClosure          *closure;
  gulong             handler_id;
  guint              after : 1;
};

enum {
  PROP_0,
  PROP_AFTER,
  PROP_OBJECT,
  PROP_RETURN_VALUE,
  PROP_SIGNAL,
  N_PROPS
};

G_DEFINE_FINAL_TYPE (FoundrySignalResponder, foundry_signal_responder, FOUNDRY_TYPE_RESPONDER)

static GParamSpec *properties[N_PROPS];

typedef struct
{
  GClosure parent_instance;
} FoundrySignalResponderClosure;

static void
foundry_signal_responder_meta_marshal (GClosure     *closure,
                                       GValue       *return_value,
                                       guint         n_param_values,
                                       const GValue *params,
                                       gpointer      invocation_hint,
                                       gpointer      marshal_data)
{
  FoundrySignalResponder *self = marshal_data;

  g_assert (FOUNDRY_IS_SIGNAL_RESPONDER (self));

  g_object_ref (self);

  if (self->return_value != NULL)
    {
      if (return_value == NULL)
        g_warning ("Return value provided but signal does not accept return value");
      else
        gtk_expression_evaluate (self->return_value, NULL, return_value);
    }

  foundry_reaction_react (FOUNDRY_REACTION (self));

  g_object_unref (self);
}

static void
foundry_signal_responder_constructed (GObject *object)
{
  FoundrySignalResponder *self = (FoundrySignalResponder *)object;
  GSignalQuery query;
  GObject *instance;
  GQuark detail;
  guint signal_id = 0;

  G_OBJECT_CLASS (foundry_signal_responder_parent_class)->constructed (object);

  if (self->detailed_signal == NULL)
    return;

  if (!(instance = g_weak_ref_get (&self->object_wr)))
    return;

  g_assert (G_IS_OBJECT (instance));

  if (!g_signal_parse_name (self->detailed_signal, G_OBJECT_TYPE (instance), &signal_id, &detail, TRUE))
    {
      g_warning ("Failed to parse detailed signal \"%s\"", self->detailed_signal);
      goto cleanup;
    }

  g_signal_query (signal_id, &query);

  self->closure = g_closure_new_simple (sizeof (FoundrySignalResponderClosure), NULL);
  g_closure_set_meta_marshal (self->closure, self, foundry_signal_responder_meta_marshal);
  g_closure_ref (self->closure);
  g_closure_sink (self->closure);

  self->handler_id = g_signal_connect_closure_by_id (instance, signal_id, detail, self->closure, self->after);

cleanup:
  g_object_unref (instance);
}

static void
foundry_signal_responder_finalize (GObject *object)
{
  FoundrySignalResponder *self = (FoundrySignalResponder *)object;
  GObject *instance;

  if ((instance = g_weak_ref_get (&self->object_wr)))
    {
      g_clear_signal_handler (&self->handler_id, instance);
      g_object_unref (instance);
    }

  if (self->closure != NULL)
    {
      g_closure_invalidate (self->closure);
      g_clear_pointer (&self->closure, g_closure_unref);
    }

  g_weak_ref_clear (&self->object_wr);
  g_clear_pointer (&self->detailed_signal, g_free);

  G_OBJECT_CLASS (foundry_signal_responder_parent_class)->finalize (object);
}

static void
foundry_signal_responder_get_property (GObject    *object,
                                       guint       prop_id,
                                       GValue     *value,
                                       GParamSpec *pspec)
{
  FoundrySignalResponder *self = FOUNDRY_SIGNAL_RESPONDER (object);

  switch (prop_id)
    {
    case PROP_AFTER:
      g_value_set_boolean (value, self->after);
      break;

    case PROP_OBJECT:
      g_value_take_object (value, g_weak_ref_get (&self->object_wr));
      break;

    case PROP_RETURN_VALUE:
      gtk_value_set_expression (value, self->return_value);
      break;

    case PROP_SIGNAL:
      g_value_set_string (value, self->detailed_signal);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_signal_responder_set_property (GObject      *object,
                                       guint         prop_id,
                                       const GValue *value,
                                       GParamSpec   *pspec)
{
  FoundrySignalResponder *self = FOUNDRY_SIGNAL_RESPONDER (object);

  switch (prop_id)
    {
    case PROP_AFTER:
      self->after = g_value_get_boolean (value);
      break;

    case PROP_OBJECT:
      g_weak_ref_set (&self->object_wr, g_value_get_object (value));
      break;

    case PROP_RETURN_VALUE:
      g_clear_pointer (&self->return_value, gtk_expression_unref);
      self->return_value = gtk_value_dup_expression (value);
      break;

    case PROP_SIGNAL:
      self->detailed_signal = g_value_dup_string (value);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_signal_responder_class_init (FoundrySignalResponderClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->constructed = foundry_signal_responder_constructed;
  object_class->finalize = foundry_signal_responder_finalize;
  object_class->get_property = foundry_signal_responder_get_property;
  object_class->set_property = foundry_signal_responder_set_property;

  properties[PROP_AFTER] =
    g_param_spec_boolean ("after", NULL, NULL,
                          FALSE,
                          (G_PARAM_READWRITE |
                           G_PARAM_CONSTRUCT_ONLY |
                           G_PARAM_STATIC_STRINGS));

  properties[PROP_OBJECT] =
    g_param_spec_object ("object", NULL, NULL,
                         G_TYPE_OBJECT,
                         (G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_RETURN_VALUE] =
    gtk_param_spec_expression ("return-value", NULL, NULL,
                               (G_PARAM_READWRITE |
                                G_PARAM_STATIC_STRINGS));

  properties[PROP_SIGNAL] =
    g_param_spec_string ("signal", NULL, NULL,
                         NULL,
                         (G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_signal_responder_init (FoundrySignalResponder *self)
{
  g_weak_ref_init (&self->object_wr, NULL);
}

FoundrySignalResponder *
foundry_signal_responder_new (GObject         *object,
                              const char      *detailed_signal,
                              gboolean         after,
                              GtkExpression   *return_value,
                              FoundryReaction *reaction)
{
  g_return_val_if_fail (G_IS_OBJECT (object), NULL);
  g_return_val_if_fail (signal != NULL, NULL);
  g_return_val_if_fail (!reaction || FOUNDRY_IS_REACTION (reaction), NULL);
  g_return_val_if_fail (!return_value || GTK_IS_EXPRESSION (return_value), NULL);

  return g_object_new (FOUNDRY_TYPE_SIGNAL_RESPONDER,
                       "after", !!after,
                       "object", object,
                       "signal", detailed_signal,
                       "reaction", reaction,
                       "return-value", return_value,
                       NULL);
}
