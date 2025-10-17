/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 18, 2023.
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
#include <iodbc.h>
#include <odbcinst.h>
#include <gtk/gtk.h>
#include <unicode.h>

#ifndef	_GTKGUI_H
#define	_GTKGUI_H

extern char* szDSNColumnNames[];
extern char* szTabNames[];
extern char* szDSNButtons[];
extern char* szDriverColumnNames[];


typedef struct TFILEDSN
{
  GtkWidget *name_entry, *mainwnd;
  char *name;
} TFILEDSN;

typedef struct TDSNCHOOSER
{
  GtkWidget *mainwnd, *udsnlist, *sdsnlist; 
  GtkWidget *uadd, *uremove, *utest, *uconfigure;
  GtkWidget *sadd, *sremove, *stest, *sconfigure;
  GtkWidget *fadd, *fremove, *ftest, *fconfigure, *fsetdir;
  GtkWidget *dir_list, *file_list, *file_entry, *dir_combo;
  wchar_t *dsn;
  wchar_t *fdsn;
  char curr_dir[1024];
  int type_dsn;
} TDSNCHOOSER;

typedef struct TDRIVERCHOOSER
{
  GtkWidget *driverlist, *mainwnd, *b_add, *b_remove, *b_configure, *b_finish;
  wchar_t *driver;
} TDRIVERCHOOSER;

typedef struct TFDRIVERCHOOSER
{
  GtkWidget *driverlist, *mainwnd;
  GtkWidget *dsn_entry, *b_back, *b_continue;
  GtkWidget *mess_entry, *tab_panel, *browse_sel;
  char *curr_dir;
  char *attrs;
  char *dsn;
  BOOL verify_conn;
  wchar_t *driver;
  BOOL ok;
} TFDRIVERCHOOSER;

typedef struct TCONNECTIONPOOLING
{
  GtkWidget *driverlist, *mainwnd, *enperfmon_rb, *disperfmon_rb,
      *retwait_entry, *timeout_entry, *probe_entry;
  BOOL changed;
  char timeout[64];
  char probe[512];
} TCONNECTIONPOOLING;

typedef struct TTRANSLATORCHOOSER
{
  GtkWidget *translatorlist, *mainwnd, *b_finish;
  wchar_t *translator;
} TTRANSLATORCHOOSER;

typedef struct TCOMPONENT
{
  GtkWidget *componentlist;
} TCOMPONENT;

typedef struct TTRACING
{
  GtkWidget *logfile_entry, *tracelib_entry, *b_start_stop;
  GtkWidget *donttrace_rb, *allthetime_rb, *onetime_rb;
  GtkWidget *filesel;
  BOOL changed;
} TTRACING;

typedef struct TCONFIRM
{
  GtkWidget *mainwnd;
  BOOL yes_no;
} TCONFIRM;

typedef struct TDRIVERSETUP
{
  GtkWidget *name_entry, *driver_entry, *setup_entry, *key_list, *bupdate;
  GtkWidget *key_entry, *value_entry;
  GtkWidget *mainwnd, *filesel;
  LPSTR connstr;
} TDRIVERSETUP;


typedef struct TGENSETUP
{
  GtkWidget *dsn_entry, *key_list, *bupdate;
  GtkWidget *key_entry, *value_entry;
  GtkWidget *mainwnd;
  GtkWidget *verify_conn_cb;
  LPSTR connstr;
  BOOL verify_conn;
} TGENSETUP;



void adddsns_to_list(GtkWidget* widget, BOOL systemDSN);
void userdsn_add_clicked(GtkWidget* widget, TDSNCHOOSER *choose_t);
void userdsn_remove_clicked(GtkWidget* widget, TDSNCHOOSER *choose_t);
void userdsn_configure_clicked(GtkWidget* widget, TDSNCHOOSER *choose_t);
void userdsn_test_clicked(GtkWidget* widget, TDSNCHOOSER *choose_t);
void systemdsn_add_clicked(GtkWidget* widget, TDSNCHOOSER *choose_t);
void systemdsn_remove_clicked(GtkWidget* widget, TDSNCHOOSER *choose_t);
void systemdsn_configure_clicked(GtkWidget* widget, TDSNCHOOSER *choose_t);
void systemdsn_test_clicked(GtkWidget* widget, TDSNCHOOSER *choose_t);
void filedsn_add_clicked(GtkWidget* widget, TDSNCHOOSER *choose_t);
void filedsn_remove_clicked(GtkWidget* widget, TDSNCHOOSER *choose_t);
void filedsn_configure_clicked(GtkWidget* widget, TDSNCHOOSER *choose_t);
void filedsn_test_clicked(GtkWidget* widget, TDSNCHOOSER *choose_t);
void filedsn_setdir_clicked(GtkWidget* widget, TDSNCHOOSER *choose_t);
void userdsn_list_select(GtkWidget* widget, gint row, gint column, GdkEvent *event, TDSNCHOOSER *choose_t);
void userdsn_list_unselect(GtkWidget* widget, gint row, gint column, GdkEvent *event, TDSNCHOOSER *choose_t);
void systemdsn_list_select(GtkWidget* widget, gint row, gint column, GdkEvent *event, TDSNCHOOSER *choose_t);
void systemdsn_list_unselect(GtkWidget* widget, gint row, gint column, GdkEvent *event, TDSNCHOOSER *choose_t);
void filedsn_filelist_select(GtkWidget* widget, gint row, gint column, GdkEvent *event, TDSNCHOOSER *choose_t);
void filedsn_filelist_unselect(GtkWidget* widget, gint row, gint column, GdkEvent *event, TDSNCHOOSER *choose_t);
void filedsn_dirlist_select(GtkWidget* widget, gint row, gint column, GdkEvent *event, TDSNCHOOSER *choose_t);
void filedsn_lookin_clicked(GtkWidget* widget, void **array);
void adddrivers_to_list(GtkWidget* widget, GtkWidget* dlg);
void addtranslators_to_list(GtkWidget* widget, GtkWidget* dlg);
void adddirectories_to_list(HWND hwnd, GtkWidget* widget, LPCSTR path);
void addfiles_to_list(HWND hwnd, GtkWidget* widget, LPCSTR path);
void addlistofdir_to_optionmenu(GtkWidget* widget, LPCSTR path, TDSNCHOOSER *choose_t);
LPSTR create_keyval (HWND wnd, LPCSTR attrs, BOOL *verify_conn);
LPSTR create_fgensetup (HWND hwnd, LPCSTR dsn, LPCSTR attrs, BOOL add, BOOL *verify_conn);

#endif
