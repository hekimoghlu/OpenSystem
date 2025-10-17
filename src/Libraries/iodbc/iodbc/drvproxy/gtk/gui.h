/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 26, 2023.
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

#ifndef	_GTKGUI_H
#define _GTKGUI_H

typedef struct TLOGIN
{
  GtkWidget *username, *password, *mainwnd;
  char *user, *pwd;
  BOOL ok;
} TLOGIN;

typedef struct TGENSETUP
{
  GtkWidget *dsn_entry, *comment_entry, *key_list, *bupdate;
  GtkWidget *key_entry, *value_entry;
  GtkWidget *mainwnd;
  LPSTR connstr;
} TGENSETUP;

typedef struct TCONFIRM
{
  GtkWidget *mainwnd;
  BOOL yes_no;
} TCONFIRM;

#endif
