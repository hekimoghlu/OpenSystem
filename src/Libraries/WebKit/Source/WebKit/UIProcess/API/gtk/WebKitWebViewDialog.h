/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 12, 2023.
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

#define WEBKIT_TYPE_WEB_VIEW_DIALOG            (webkit_web_view_dialog_get_type())
#define WEBKIT_WEB_VIEW_DIALOG(obj)            (G_TYPE_CHECK_INSTANCE_CAST((obj), WEBKIT_TYPE_WEB_VIEW_DIALOG, WebKitWebViewDialog))
#define WEBKIT_IS_WEB_VIEW_DIALOG(obj)         (G_TYPE_CHECK_INSTANCE_TYPE((obj), WEBKIT_TYPE_WEB_VIEW_DIALOG))
#define WEBKIT_WEB_VIEW_DIALOG_CLASS(klass)    (G_TYPE_CHECK_CLASS_CAST((klass),  WEBKIT_TYPE_WEB_VIEW_DIALOG, WebKitWebViewDialogClass))
#define WEBKIT_IS_WEB_VIEW_DIALOG_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass),  WEBKIT_TYPE_WEB_VIEW_DIALOG))
#define WEBKIT_WEB_VIEW_DIALOG_GET_CLASS(obj)  (G_TYPE_INSTANCE_GET_CLASS((obj),  WEBKIT_TYPE_WEB_VIEW_DIALOG, WebKitWebViewDialogClass))

typedef struct _WebKitWebViewDialog        WebKitWebViewDialog;
typedef struct _WebKitWebViewDialogClass   WebKitWebViewDialogClass;
typedef struct _WebKitWebViewDialogPrivate WebKitWebViewDialogPrivate;

struct _WebKitWebViewDialog {
#if USE(GTK4)
    GtkWidget parent;
#else
    GtkEventBox parent;
#endif

    /*< private >*/
    WebKitWebViewDialogPrivate* priv;
};

struct _WebKitWebViewDialogClass {
#if USE(GTK4)
    GtkWidgetClass parentClass;
#else
    GtkEventBoxClass parentClass;
#endif
};

GType webkit_web_view_dialog_get_type();
#if USE(GTK4)
void webkitWebViewDialogSetChild(WebKitWebViewDialog*, GtkWidget*);
GtkWidget* webkitWebViewDialogGetChild(WebKitWebViewDialog*);
#endif

G_END_DECLS
