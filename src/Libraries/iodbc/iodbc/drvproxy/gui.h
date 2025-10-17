/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 8, 2022.
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

#if defined(__BEOS__)
#include "be/gui.h"
#elif defined(macintosh)
#include "mac/gui.h"
#elif defined(__GTK__)
#include "gtk/gui.h"
#elif defined(__QT__)
#include "qt/gui.h"
#elif defined(__APPLE__)
#include "macosx/gui.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

#ifndef	_GUI_H
#define _GUI_H

LPSTR create_gensetup (HWND hwnd, LPCSTR dsn, LPCSTR attrs, BOOL add);
void create_login (HWND hwnd, LPCSTR username, LPCSTR password, LPCSTR dsn, TLOGIN *log_t);
BOOL create_confirm (HWND hwnd, LPCSTR dsn, LPCSTR text);

#ifdef __cplusplus
}
#endif

#endif
