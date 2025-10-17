/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 3, 2024.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */
#include "gui.h"


static gint
delete_event (GtkWidget *widget, GdkEvent *event)
{
  return FALSE;
}


static void
message_ok_clicked (GtkWidget *widget, GtkWidget *message)
{
  gtk_signal_disconnect_by_func (GTK_OBJECT (message),
      GTK_SIGNAL_FUNC (gtk_main_quit), NULL);
  gtk_main_quit ();
  gtk_widget_destroy (message);
}


void
create_message (HWND hwnd, LPCSTR dsn, LPCSTR text)
{
  GtkWidget *message, *dialog_vbox1, *hbox1, *vbox1;
  GtkWidget *l_message, *dialog_action_area1, *hbuttonbox1, *b_ok;
  GtkAccelGroup *accel_group;
  guint b_ok_key;
  char msg[1024];

  if (hwnd == NULL || !GTK_IS_WIDGET (hwnd))
    return;

  accel_group = gtk_accel_group_new ();

  message = gtk_dialog_new ();
  if (dsn)
    sprintf (msg, "Message on DSN %s", dsn);
  else
    sprintf (msg, "Message ...");
  gtk_object_set_data (GTK_OBJECT (message), "message", message);
  gtk_window_set_title (GTK_WINDOW (message), msg);
  gtk_window_set_position (GTK_WINDOW (message), GTK_WIN_POS_CENTER);
  gtk_window_set_modal (GTK_WINDOW (message), TRUE);
#if GTK_CHECK_VERSION(2,0,0)
  gtk_window_set_policy (GTK_WINDOW (message), FALSE, FALSE, FALSE);
#else
  gtk_window_set_policy (GTK_WINDOW (message), TRUE, TRUE, FALSE);
#endif

#if GTK_CHECK_VERSION(2,0,0)
  gtk_widget_show (message);
#endif

  dialog_vbox1 = GTK_DIALOG (message)->vbox;
  gtk_object_set_data (GTK_OBJECT (message), "dialog_vbox1", dialog_vbox1);
  gtk_widget_show (dialog_vbox1);

  hbox1 = gtk_hbox_new (FALSE, 6);
  gtk_widget_ref (hbox1);
  gtk_object_set_data_full (GTK_OBJECT (message), "hbox1", hbox1,
      (GtkDestroyNotify) gtk_widget_unref);
  gtk_widget_show (hbox1);
  gtk_box_pack_start (GTK_BOX (dialog_vbox1), hbox1, TRUE, TRUE, 0);
  gtk_container_set_border_width (GTK_CONTAINER (hbox1), 6);

  vbox1 = gtk_vbox_new (TRUE, 0);
  gtk_widget_ref (vbox1);
  gtk_object_set_data_full (GTK_OBJECT (message), "vbox1", vbox1,
      (GtkDestroyNotify) gtk_widget_unref);
  gtk_widget_show (vbox1);
  gtk_box_pack_start (GTK_BOX (hbox1), vbox1, TRUE, TRUE, 0);

  l_message = gtk_label_new ("");
  gtk_label_parse_uline (GTK_LABEL (l_message), (text) ? text : "");
  gtk_widget_ref (l_message);
  gtk_object_set_data_full (GTK_OBJECT (message), "l_message", l_message,
      (GtkDestroyNotify) gtk_widget_unref);
  gtk_widget_show (l_message);
  gtk_box_pack_start (GTK_BOX (vbox1), l_message, FALSE, TRUE, 0);
  gtk_label_set_justify (GTK_LABEL (l_message), GTK_JUSTIFY_LEFT);
  gtk_label_set_line_wrap (GTK_LABEL (l_message), TRUE);

  dialog_action_area1 = GTK_DIALOG (message)->action_area;
  gtk_object_set_data (GTK_OBJECT (message), "dialog_action_area1",
      dialog_action_area1);
  gtk_widget_show (dialog_action_area1);
  gtk_container_set_border_width (GTK_CONTAINER (dialog_action_area1), 5);

  hbuttonbox1 = gtk_hbutton_box_new ();
  gtk_widget_ref (hbuttonbox1);
  gtk_object_set_data_full (GTK_OBJECT (message), "hbuttonbox1", hbuttonbox1,
      (GtkDestroyNotify) gtk_widget_unref);
  gtk_widget_show (hbuttonbox1);
  gtk_box_pack_start (GTK_BOX (dialog_action_area1), hbuttonbox1, TRUE, TRUE,
      0);
  gtk_button_box_set_layout (GTK_BUTTON_BOX (hbuttonbox1), GTK_BUTTONBOX_END);
  gtk_button_box_set_spacing (GTK_BUTTON_BOX (hbuttonbox1), 10);

  b_ok = gtk_button_new_with_label ("");
  b_ok_key = gtk_label_parse_uline (GTK_LABEL (GTK_BIN (b_ok)->child), "_Ok");
  gtk_widget_add_accelerator (b_ok, "clicked", accel_group,
      b_ok_key, GDK_MOD1_MASK, 0);
  gtk_widget_ref (b_ok);
  gtk_object_set_data_full (GTK_OBJECT (message), "b_ok", b_ok,
      (GtkDestroyNotify) gtk_widget_unref);
  gtk_widget_show (b_ok);
  gtk_container_add (GTK_CONTAINER (hbuttonbox1), b_ok);
  GTK_WIDGET_SET_FLAGS (b_ok, GTK_CAN_DEFAULT);

  /* Ok button events */
  gtk_signal_connect (GTK_OBJECT (b_ok), "clicked",
      GTK_SIGNAL_FUNC (message_ok_clicked), message);
  /* Close window button events */
  gtk_signal_connect (GTK_OBJECT (message), "delete_event",
      GTK_SIGNAL_FUNC (delete_event), NULL);
  gtk_signal_connect (GTK_OBJECT (message), "destroy",
      GTK_SIGNAL_FUNC (gtk_main_quit), NULL);

  gtk_window_add_accel_group (GTK_WINDOW (message), accel_group);

  gtk_widget_show_all (message);
  gtk_main ();
}

void
create_messagew (HWND hwnd, LPCWSTR dsn, LPCWSTR text)
{
  LPSTR _dsn = NULL;
  LPSTR _text = NULL;

  _dsn = dm_SQL_WtoU8(dsn, SQL_NTS);
  _text = dm_SQL_WtoU8(text, SQL_NTS);

  create_message(hwnd, _dsn, _text);

  if (_dsn)
    free(_dsn);
  if (_text)
    free(_text);
}
