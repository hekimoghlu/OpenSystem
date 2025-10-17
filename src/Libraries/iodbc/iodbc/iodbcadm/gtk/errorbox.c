/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 14, 2025.
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
#include "error.xpm"


static gint
delete_event (GtkWidget *widget, GdkEvent *event)
{
  return FALSE;
}


static void
error_ok_clicked (GtkWidget *widget, GtkWidget *error)
{
  gtk_signal_disconnect_by_func (GTK_OBJECT (error),
      GTK_SIGNAL_FUNC (gtk_main_quit), NULL);
  gtk_main_quit ();
  gtk_widget_destroy (error);
}


void
create_error (HWND hwnd, LPCSTR dsn, LPCSTR text, LPCSTR errmsg)
{
  GtkWidget *error, *dialog_vbox1, *hbox1, *pixmap1, *vbox1;
  GtkWidget *l_text, *l_error, *dialog_action_area1, *hbuttonbox1, *b_ok;
  GtkAccelGroup *accel_group;
  GdkPixmap *pixmap;
  GdkBitmap *mask;
  GtkStyle *style;
  guint b_ok_key;
  char msg[1024];

  if (hwnd == NULL || !GTK_IS_WIDGET (hwnd))
    return;

  accel_group = gtk_accel_group_new ();

  error = gtk_dialog_new ();
  if (dsn)
    sprintf (msg, "Error : %s", dsn);
  else
    sprintf (msg, "Error ...");

  gtk_object_set_data (GTK_OBJECT (error), "error", error);
  gtk_window_set_title (GTK_WINDOW (error), msg);
  gtk_window_set_position (GTK_WINDOW (error), GTK_WIN_POS_CENTER);
  gtk_window_set_modal (GTK_WINDOW (error), TRUE);
  gtk_window_set_policy (GTK_WINDOW (error), FALSE, FALSE, FALSE);

#if GTK_CHECK_VERSION(2,0,0)
  gtk_widget_show (error);
#endif

  dialog_vbox1 = GTK_DIALOG (error)->vbox;
  gtk_object_set_data (GTK_OBJECT (error), "dialog_vbox1", dialog_vbox1);
  gtk_widget_show (dialog_vbox1);

  hbox1 = gtk_hbox_new (FALSE, 6);
  gtk_widget_ref (hbox1);
  gtk_object_set_data_full (GTK_OBJECT (error), "hbox1", hbox1,
      (GtkDestroyNotify) gtk_widget_unref);
  gtk_widget_show (hbox1);
  gtk_box_pack_start (GTK_BOX (dialog_vbox1), hbox1, TRUE, TRUE, 0);
  gtk_container_set_border_width (GTK_CONTAINER (hbox1), 6);

#if GTK_CHECK_VERSION(2,0,0)
  style = gtk_widget_get_style (error);
  pixmap =
      gdk_pixmap_create_from_xpm_d (error->window, &mask,
      &style->bg[GTK_STATE_NORMAL], (gchar **) error_xpm);
#else
  style = gtk_widget_get_style (GTK_WIDGET (hwnd));
  pixmap =
      gdk_pixmap_create_from_xpm_d (GTK_WIDGET (hwnd)->window, &mask,
      &style->bg[GTK_STATE_NORMAL], (gchar **) error_xpm);
#endif

  pixmap1 = gtk_pixmap_new (pixmap, mask);
  gtk_widget_ref (pixmap1);
  gtk_object_set_data_full (GTK_OBJECT (error), "pixmap1", pixmap1,
      (GtkDestroyNotify) gtk_widget_unref);
  gtk_widget_show (pixmap1);
  gtk_box_pack_start (GTK_BOX (hbox1), pixmap1, FALSE, FALSE, 0);

  vbox1 = gtk_vbox_new (TRUE, 0);
  gtk_widget_ref (vbox1);
  gtk_object_set_data_full (GTK_OBJECT (error), "vbox1", vbox1,
      (GtkDestroyNotify) gtk_widget_unref);
  gtk_widget_show (vbox1);
  gtk_box_pack_start (GTK_BOX (hbox1), vbox1, TRUE, TRUE, 0);

  l_text = gtk_label_new ("");
  gtk_label_parse_uline (GTK_LABEL (l_text), (text) ? text : "");
  gtk_widget_ref (l_text);
  gtk_object_set_data_full (GTK_OBJECT (error), "l_text", l_text,
      (GtkDestroyNotify) gtk_widget_unref);
  gtk_widget_show (l_text);
  gtk_box_pack_start (GTK_BOX (vbox1), l_text, FALSE, TRUE, 0);
  gtk_label_set_justify (GTK_LABEL (l_text), GTK_JUSTIFY_LEFT);
  gtk_label_set_line_wrap (GTK_LABEL (l_text), TRUE);

  l_error = gtk_label_new ("");
  gtk_label_parse_uline (GTK_LABEL (l_error), (errmsg) ? errmsg : "");
  gtk_widget_ref (l_error);
  gtk_object_set_data_full (GTK_OBJECT (error), "l_error", l_error,
      (GtkDestroyNotify) gtk_widget_unref);
  gtk_widget_show (l_error);
  gtk_box_pack_start (GTK_BOX (vbox1), l_error, FALSE, TRUE, 0);
  gtk_label_set_justify (GTK_LABEL (l_error), GTK_JUSTIFY_LEFT);
  gtk_label_set_line_wrap (GTK_LABEL (l_error), TRUE);

  dialog_action_area1 = GTK_DIALOG (error)->action_area;
  gtk_object_set_data (GTK_OBJECT (error), "dialog_action_area1",
      dialog_action_area1);
  gtk_widget_show (dialog_action_area1);
  gtk_container_set_border_width (GTK_CONTAINER (dialog_action_area1), 5);

  hbuttonbox1 = gtk_hbutton_box_new ();
  gtk_widget_ref (hbuttonbox1);
  gtk_object_set_data_full (GTK_OBJECT (error), "hbuttonbox1", hbuttonbox1,
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
  gtk_object_set_data_full (GTK_OBJECT (error), "b_ok", b_ok,
      (GtkDestroyNotify) gtk_widget_unref);
  gtk_widget_show (b_ok);
  gtk_container_add (GTK_CONTAINER (hbuttonbox1), b_ok);
  GTK_WIDGET_SET_FLAGS (b_ok, GTK_CAN_DEFAULT);

  /* Ok button events */
  gtk_signal_connect (GTK_OBJECT (b_ok), "clicked",
      GTK_SIGNAL_FUNC (error_ok_clicked), error);
  /* Close window button events */
  gtk_signal_connect (GTK_OBJECT (error), "delete_event",
      GTK_SIGNAL_FUNC (delete_event), NULL);
  gtk_signal_connect (GTK_OBJECT (error), "destroy",
      GTK_SIGNAL_FUNC (gtk_main_quit), NULL);

  gtk_window_add_accel_group (GTK_WINDOW (error), accel_group);

  gtk_widget_show_all (error);
  gtk_main ();
}


void
create_errorw (HWND hwnd, LPCWSTR dsn, LPCWSTR text, LPCWSTR errmsg)
{
  LPSTR _dsn = NULL;
  LPSTR _text = NULL;
  LPSTR _errmsg = NULL;

  _dsn = dm_SQL_WtoU8(dsn, SQL_NTS);
  _text = dm_SQL_WtoU8(text, SQL_NTS);
  _errmsg = dm_SQL_WtoU8(errmsg, SQL_NTS);

  create_error(hwnd, _dsn, _text, _errmsg);

  if (_dsn)
    free(_dsn);
  if (_text)
    free(_text);
  if (_errmsg)
    free(_errmsg);
}

