/*
 * Copyright (C) 2021 Purism SPC
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 *
 * Author: Thomas Booker <tw.booker@outlook.com>
 *
 * Based on calls-new-call-box by
 * Adrien Plazas <adrien.plazas@puri.sm>
 */

#include "cui-config.h"

#include "cui-dialpad.h"
#include "cui-keypad.h"

#include <adwaita.h>
#include <glib.h>
#include <glib/gi18n-lib.h>
#include <libcallaudio.h>

#define IS_NULL_OR_EMPTY(x)  ((x) == NULL || (x)[0] == '\0')

/**
 * CuiDialpad:
 *
 * A simple dial pad to enter phone numbers.
 *
 */

enum {
  PROP_0,
  PROP_NUMBER,
  PROP_LAST_PROP,
};
static GParamSpec *props[PROP_LAST_PROP];

enum {
  DIALED = 0,
  LAST_SIGNAL,
};
static guint signals[LAST_SIGNAL];


struct _CuiDialpad {
  AdwBin               parent_instance;

  AdwClamp            *clamp;
  GtkEntry            *keypad_entry;
  CuiKeypad           *keypad;
  GtkButton           *dial;
  GtkButton           *backspace;
};

G_DEFINE_TYPE (CuiDialpad, cui_dialpad, ADW_TYPE_BIN);

static void
cui_dialpad_get_property (GObject    *object,
                          guint       property_id,
                          GValue     *value,
                          GParamSpec *pspec)
{
  CuiDialpad *self = CUI_DIALPAD (object);

  switch (property_id) {
  case PROP_NUMBER:
    g_value_set_string (value, cui_dialpad_get_number (self));
    break;
  default:
    G_OBJECT_WARN_INVALID_PROPERTY_ID (object, property_id, pspec);
    break;
  }
}

static void
cui_dialpad_set_property (GObject      *object,
                          guint         property_id,
                          const GValue *value,
                          GParamSpec   *pspec)
{
  CuiDialpad *self = CUI_DIALPAD (object);

  switch (property_id) {
  case PROP_NUMBER:
    cui_dialpad_set_number (self, g_value_get_string (value));
    break;
  default:
    G_OBJECT_WARN_INVALID_PROPERTY_ID (object, property_id, pspec);
    break;
  }
}

/* this handler is used both when the dial button is clicked or "enter" pressed on the entry */
static void
dial_clicked_or_activated_cb (CuiDialpad *self)
{
  GtkEntry *entry = cui_keypad_get_entry (self->keypad);
  const char *text = gtk_editable_get_text (GTK_EDITABLE (entry));

  g_signal_emit (self, signals[DIALED], 0, text);
}

static void
backspace_clicked_cb (CuiDialpad *self)
{
  GtkEntry *entry = cui_keypad_get_entry (self->keypad);
  const char* text = gtk_editable_get_text (GTK_EDITABLE (entry));
  unsigned long len = strlen(text);

  if (len > 0) {
    gtk_editable_delete_text (GTK_EDITABLE (entry), len - 1, len);
  }
}

static void
long_press_backspace_cb (CuiDialpad *self)
{
  GtkEntry *entry = cui_keypad_get_entry (self->keypad);

  gtk_editable_delete_text (GTK_EDITABLE (entry), 0, -1);
}


static void
cui_dialpad_dispose (GObject *object)
{
  CuiDialpad *self = CUI_DIALPAD (object);

  GtkWidget *clamp = GTK_WIDGET (self->clamp);
  g_clear_pointer (&clamp, gtk_widget_unparent);

  G_OBJECT_CLASS (cui_dialpad_parent_class)->dispose (object);
}


static void
cui_dialpad_class_init (CuiDialpadClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  GtkWidgetClass *widget_class = GTK_WIDGET_CLASS (klass);

  object_class->get_property = cui_dialpad_get_property;
  object_class->set_property = cui_dialpad_set_property;

  object_class->dispose = cui_dialpad_dispose;

  signals[DIALED] =
    g_signal_new ("dialed",
                  G_TYPE_FROM_CLASS (klass),
                  G_SIGNAL_RUN_LAST,
                  0,
                  NULL, NULL, NULL,
                  G_TYPE_NONE,
                  1,
                  G_TYPE_STRING);

  /**
   * CuiDialpad:number:
   *
   * The current number in the dialpad
   */
  props[PROP_NUMBER] = g_param_spec_string ("number",
                                            "phone number",
                                            "phone number to dial",
                                            "",
                                            G_PARAM_READWRITE);


  g_object_class_install_properties (object_class, PROP_LAST_PROP, props);

  gtk_widget_class_set_template_from_resource (widget_class, "/org/gnome/CallUI/ui/cui-dialpad.ui");

  gtk_widget_class_bind_template_child (widget_class, CuiDialpad, clamp);
  gtk_widget_class_bind_template_child (widget_class, CuiDialpad, keypad);
  gtk_widget_class_bind_template_child (widget_class, CuiDialpad, keypad_entry);
  gtk_widget_class_bind_template_child (widget_class, CuiDialpad, dial);
  gtk_widget_class_bind_template_child (widget_class, CuiDialpad, backspace);
  gtk_widget_class_bind_template_callback (widget_class, dial_clicked_or_activated_cb);
  gtk_widget_class_bind_template_callback (widget_class, backspace_clicked_cb);
  gtk_widget_class_bind_template_callback (widget_class, long_press_backspace_cb);

  gtk_widget_class_set_css_name (widget_class, "cui-dialpad");
}


static void
cui_dialpad_init (CuiDialpad *self)
{
  GtkGesture *gesture = gtk_gesture_long_press_new ();
  g_signal_connect_swapped (gesture, "pressed", G_CALLBACK (long_press_backspace_cb), self);

  gtk_widget_init_template (GTK_WIDGET (self));

  gtk_widget_add_controller (GTK_WIDGET (self->backspace), GTK_EVENT_CONTROLLER (gesture));
}

/**
 * cui_dialpad_new:
 *
 * Creates a new #CuiDialpad.
 * Returns: the new #CuiDialpad
 */
CuiDialpad *
cui_dialpad_new (void)
{
  return g_object_new (CUI_TYPE_DIALPAD,
                       NULL);
}

/**
 * cui_dialpad_get_number:
 * @self: The dialpad
 *
 * Gets the current number in the dialpad
 *
 * Returns: the current number
 */
const char *
cui_dialpad_get_number (CuiDialpad *self)
{
  g_return_val_if_fail (CUI_IS_DIALPAD (self), NULL);

  return gtk_editable_get_text (GTK_EDITABLE (self->keypad_entry));
}

/**
 * cui_dialpad_set_number:
 * @self: The dialpad
 * @number: The number to set
 *
 * Sets the current number in the dialpad
 */
void
cui_dialpad_set_number (CuiDialpad *self, const char *number)
{
  g_return_if_fail (CUI_IS_DIALPAD (self));

  gtk_editable_set_text (GTK_EDITABLE (self->keypad_entry), number);
}
