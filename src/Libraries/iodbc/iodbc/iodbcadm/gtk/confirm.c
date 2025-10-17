/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 30, 2024.
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
#include <odbcinst.h>
#include <unicode.h>
#include "gui.h"
#include "question.xpm"


static void
confirm_yes_clicked (GtkWidget *widget, TCONFIRM *confirm_t)
{
  if (confirm_t)
    {
      confirm_t->yes_no = TRUE;

      gtk_signal_disconnect_by_func (GTK_OBJECT (confirm_t->mainwnd),
	  GTK_SIGNAL_FUNC (gtk_main_quit), NULL);
      gtk_main_quit ();
      gtk_widget_destroy (confirm_t->mainwnd);
    }
}


static void
confirm_no_clicked (GtkWidget *widget, TCONFIRM *confirm_t)
{
  if (confirm_t)
    {
      confirm_t->yes_no = FALSE;

      gtk_signal_disconnect_by_func (GTK_OBJECT (confirm_t->mainwnd),
	  GTK_SIGNAL_FUNC (gtk_main_quit), NULL);
      gtk_main_quit ();
      gtk_widget_destroy (confirm_t->mainwnd);
    }
}


static gint
delete_event (GtkWidget *widget, GdkEvent *event, TCONFIRM *confirm_t)
{
  confirm_no_clicked (widget, confirm_t);

  return FALSE;
}


BOOL
create_confirm (HWND hwnd, LPCSTR dsn, LPCSTR text)
{
  GtkWidget *confirm, *dialog_vbox1, *hbox1, *pixmap1, *l_text;
  GtkWidget *dialog_action_area1, *hbuttonbox1, *b_yes, *b_no;
  guint b_yes_key, b_no_key;
  GdkPixmap *pixmap;
  GdkBitmap *mask;
  GtkStyle *style;
  GtkAccelGroup *accel_group;
  char msg[1024];
  TCONFIRM confirm_t;

  if (hwnd == NULL || !GTK_IS_WIDGET (hwnd))
    return FALSE;

  accel_group = gtk_accel_group_new ();

  confirm = gtk_dialog_new ();
  if (dsn)
    sprintf (msg, "Confirm action/operation on %s", dsn);
  else
    sprintf (msg, "Confirm action/operation ...");
  gtk_object_set_data (GTK_OBJECT (confirm), "confirm", confirm);
  gtk_window_set_title (GTK_WINDOW (confirm), msg);
  gtk_window_set_position (GTK_WINDOW (confirm), GTK_WIN_POS_CENTER);
  gtk_window_set_modal (GTK_WINDOW (confirm), TRUE);
  gtk_window_set_policy (GTK_WINDOW (confirm), FALSE, FALSE, FALSE);

#if GTK_CHECK_VERSION(2,0,0)
  gtk_widget_show (confirm);
#endif

  dialog_vbox1 = GTK_DIALOG (confirm)->vbox;
  gtk_object_set_data (GTK_OBJECT (confirm), "dialog_vbox1", dialog_vbox1);
  gtk_widget_show (dialog_vbox1);

  hbox1 = gtk_hbox_new (FALSE, 6);
  gtk_widget_ref (hbox1);
  gtk_object_set_data_full (GTK_OBJECT (confirm), "hbox1", hbox1,
      (GtkDestroyNotify) gtk_widget_unref);
  gtk_widget_show (hbox1);
  gtk_box_pack_start (GTK_BOX (dialog_vbox1), hbox1, TRUE, TRUE, 0);
  gtk_container_set_border_width (GTK_CONTAINER (hbox1), 6);

#if GTK_CHECK_VERSION(2,0,0)
  style = gtk_widget_get_style (confirm);
  pixmap =
      gdk_pixmap_create_from_xpm_d (confirm->window, &mask,
      &style->bg[GTK_STATE_NORMAL], (gchar **) question_xpm);
#else
  style = gtk_widget_get_style (GTK_WIDGET (hwnd));
  pixmap =
      gdk_pixmap_create_from_xpm_d (GTK_WIDGET (hwnd)->window, &mask,
      &style->bg[GTK_STATE_NORMAL], (gchar **) question_xpm);
#endif

  pixmap1 = gtk_pixmap_new (pixmap, mask);
  gtk_widget_ref (pixmap1);
  gtk_object_set_data_full (GTK_OBJECT (confirm), "pixmap1", pixmap1,
      (GtkDestroyNotify) gtk_widget_unref);
  gtk_widget_show (pixmap1);
  gtk_box_pack_start (GTK_BOX (hbox1), pixmap1, FALSE, FALSE, 0);

  l_text = gtk_label_new ("");
  gtk_label_parse_uline (GTK_LABEL (l_text), text);
  gtk_widget_ref (l_text);
  gtk_object_set_data_full (GTK_OBJECT (confirm), "l_text", l_text,
      (GtkDestroyNotify) gtk_widget_unref);
  gtk_widget_show (l_text);
  gtk_box_pack_start (GTK_BOX (hbox1), l_text, TRUE, TRUE, 0);
  gtk_label_set_justify (GTK_LABEL (l_text), GTK_JUSTIFY_LEFT);
  gtk_label_set_line_wrap (GTK_LABEL (l_text), TRUE);

  dialog_action_area1 = GTK_DIALOG (confirm)->action_area;
  gtk_object_set_data (GTK_OBJECT (confirm), "dialog_action_area1",
      dialog_action_area1);
  gtk_widget_show (dialog_action_area1);
  gtk_container_set_border_width (GTK_CONTAINER (dialog_action_area1), 5);

  hbuttonbox1 = gtk_hbutton_box_new ();
  gtk_widget_ref (hbuttonbox1);
  gtk_object_set_data_full (GTK_OBJECT (confirm), "hbuttonbox1", hbuttonbox1,
      (GtkDestroyNotify) gtk_widget_unref);
  gtk_widget_show (hbuttonbox1);
  gtk_box_pack_start (GTK_BOX (dialog_action_area1), hbuttonbox1, TRUE, TRUE,
      0);
  gtk_button_box_set_layout (GTK_BUTTON_BOX (hbuttonbox1), GTK_BUTTONBOX_END);
  gtk_button_box_set_spacing (GTK_BUTTON_BOX (hbuttonbox1), 10);

  b_yes = gtk_button_new_with_label ("");
  b_yes_key = gtk_label_parse_uline (GTK_LABEL (GTK_BIN (b_yes)->child),
      "_Yes");
  gtk_widget_add_accelerator (b_yes, "clicked", accel_group,
      b_yes_key, GDK_MOD1_MASK, 0);
  gtk_widget_ref (b_yes);
  gtk_object_set_data_full (GTK_OBJECT (confirm), "b_yes", b_yes,
      (GtkDestroyNotify) gtk_widget_unref);
  gtk_widget_show (b_yes);
  gtk_container_add (GTK_CONTAINER (hbuttonbox1), b_yes);
  GTK_WIDGET_SET_FLAGS (b_yes, GTK_CAN_DEFAULT);

  b_no = gtk_button_new_with_label ("");
  b_no_key = gtk_label_parse_uline (GTK_LABEL (GTK_BIN (b_no)->child), "_No");
  gtk_widget_add_accelerator (b_no, "clicked", accel_group,
      b_no_key, GDK_MOD1_MASK, 0);
  gtk_widget_ref (b_no);
  gtk_object_set_data_full (GTK_OBJECT (confirm), "b_no", b_no,
      (GtkDestroyNotify) gtk_widget_unref);
  gtk_widget_show (b_no);
  gtk_container_add (GTK_CONTAINER (hbuttonbox1), b_no);
  GTK_WIDGET_SET_FLAGS (b_no, GTK_CAN_DEFAULT);

  /* Yes button events */
  gtk_signal_connect (GTK_OBJECT (b_yes), "clicked",
      GTK_SIGNAL_FUNC (confirm_yes_clicked), &confirm_t);
  /* No button events */
  gtk_signal_connect (GTK_OBJECT (b_no), "clicked",
      GTK_SIGNAL_FUNC (confirm_no_clicked), &confirm_t);
  /* Close window button events */
  gtk_signal_connect (GTK_OBJECT (confirm), "delete_event",
      GTK_SIGNAL_FUNC (delete_event), &confirm_t);
  gtk_signal_connect (GTK_OBJECT (confirm), "destroy",
      GTK_SIGNAL_FUNC (gtk_main_quit), NULL);

  gtk_window_add_accel_group (GTK_WINDOW (confirm), accel_group);

  confirm_t.yes_no = FALSE;
  confirm_t.mainwnd = confirm;

  gtk_widget_show_all (confirm);
  gtk_main ();

  return confirm_t.yes_no;
}

BOOL
create_confirmw (HWND hwnd, LPCWSTR dsn, LPCWSTR text)
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
