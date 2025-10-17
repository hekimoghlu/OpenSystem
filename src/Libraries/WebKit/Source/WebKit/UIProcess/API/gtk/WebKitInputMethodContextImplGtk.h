/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 15, 2024.
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

#include "WebKitInputMethodContext.h"
#include <gtk/gtk.h>

G_BEGIN_DECLS

#define WEBKIT_TYPE_INPUT_METHOD_CONTEXT_IMPL_GTK            (webkit_input_method_context_impl_gtk_get_type())
#define WEBKIT_INPUT_METHOD_CONTEXT_IMPL_GTK(obj)            (G_TYPE_CHECK_INSTANCE_CAST((obj), WEBKIT_TYPE_INPUT_METHOD_CONTEXT_IMPL_GTK, WebKitInputMethodContextImplGtk))
#define WEBKIT_IS_INPUT_METHOD_CONTEXT_IMPL_GTK(obj)         (G_TYPE_CHECK_INSTANCE_TYPE((obj), WEBKIT_TYPE_INPUT_METHOD_CONTEXT_IMPL_GTK))
#define WEBKIT_INPUT_METHOD_CONTEXT_IMPL_GTK_CLASS(klass)    (G_TYPE_CHECK_CLASS_CAST((klass),  WEBKIT_TYPE_INPUT_METHOD_CONTEXT_IMPL_GTK, WebKitInputMethodContextImplGtkClass))
#define WEBKIT_IS_INPUT_METHOD_CONTEXT_IMPL_GTK_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass),  WEBKIT_TYPE_INPUT_METHOD_CONTEXT_IMPL_GTK))
#define WEBKIT_INPUT_METHOD_CONTEXT_IMPL_GTK_GET_CLASS(obj)  (G_TYPE_INSTANCE_GET_CLASS((obj),  WEBKIT_TYPE_INPUT_METHOD_CONTEXT_IMPL_GTK, WebKitInputMethodContextImplGtkClass))

typedef struct _WebKitInputMethodContextImplGtk        WebKitInputMethodContextImplGtk;
typedef struct _WebKitInputMethodContextImplGtkClass   WebKitInputMethodContextImplGtkClass;
typedef struct _WebKitInputMethodContextImplGtkPrivate WebKitInputMethodContextImplGtkPrivate;

struct _WebKitInputMethodContextImplGtk {
    WebKitInputMethodContext parent;

    WebKitInputMethodContextImplGtkPrivate* priv;
};

struct _WebKitInputMethodContextImplGtkClass {
    WebKitInputMethodContextClass parentClass;
};

GType webkit_input_method_context_impl_gtk_get_type();
WebKitInputMethodContext* webkitInputMethodContextImplGtkNew();
void webkitInputMethodContextImplGtkSetClientWidget(WebKitInputMethodContextImplGtk*, GtkWidget*);

G_END_DECLS

