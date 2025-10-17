/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 13, 2025.
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
#pragma once

#include <gtk/gtk.h>

G_BEGIN_DECLS

#define WEBKIT_TYPE_INSPECTOR_WINDOW            (webkit_inspector_window_get_type())
#define WEBKIT_INSPECTOR_WINDOW(obj)            (G_TYPE_CHECK_INSTANCE_CAST((obj), WEBKIT_TYPE_INSPECTOR_WINDOW, WebKitInspectorWindow))
#define WEBKIT_IS_INSPECTOR_WINDOW(obj)         (G_TYPE_CHECK_INSTANCE_TYPE((obj), WEBKIT_TYPE_INSPECTOR_WINDOW))
#define WEBKIT_INSPECTOR_WINDOW_CLASS(klass)    (G_TYPE_CHECK_CLASS_CAST((klass),  WEBKIT_TYPE_INSPECTOR_WINDOW, WebKitInspectorWindowClass))
#define WEBKIT_IS_INSPECTOR_WINDOW_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass),  WEBKIT_TYPE_INSPECTOR_WINDOW))
#define WEBKIT_INSPECTOR_WINDOW_GET_CLASS(obj)  (G_TYPE_INSTANCE_GET_CLASS((obj),  WEBKIT_TYPE_INSPECTOR_WINDOW, WebKitInspectorWindowClass))

typedef struct _WebKitInspectorWindow WebKitInspectorWindow;
typedef struct _WebKitInspectorWindowClass WebKitInspectorWindowClass;

GType webkit_inspector_window_get_type(void);

GtkWidget* webkitInspectorWindowNew();
void webkitInspectorWindowSetSubtitle(WebKitInspectorWindow*, const char* subtitle);

G_END_DECLS
