/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 4, 2024.
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

#if USE(ATK)

#include <atk/atk.h>

G_BEGIN_DECLS

#define WEBKIT_TYPE_WEB_VIEW_ACCESSIBLE              (webkit_web_view_accessible_get_type())
#define WEBKIT_WEB_VIEW_ACCESSIBLE(object)           (G_TYPE_CHECK_INSTANCE_CAST((object), WEBKIT_TYPE_WEB_VIEW_ACCESSIBLE, WebKitWebViewAccessible))
#define WEBKIT_WEB_VIEW_ACCESSIBLE_CLASS(klass)      (G_TYPE_CHECK_CLASS_CAST((klass), WEBKIT_TYPE_WEB_VIEW_ACCESSIBLE, WebKitWebViewAccessibleClass))
#define WEBKIT_IS_WEB_VIEW_ACCESSIBLE(object)        (G_TYPE_CHECK_INSTANCE_TYPE((object), WEBKIT_TYPE_WEB_VIEW_ACCESSIBLE))
#define WEBKIT_IS_WEB_VIEW_ACCESSIBLE_CLASS(klass)   (G_TYPE_CHECK_CLASS_TYPE((klass), WEBKIT_TYPE_WEB_VIEW_ACCESSIBLE))
#define WEBKIT_WEB_VIEW_ACCESSIBLE_GET_CLASS(object) (G_TYPE_INSTANCE_GET_CLASS((object), WEBKIT_TYPE_WEB_VIEW_ACCESSIBLE, WebKitWebViewAccessibleClass))

typedef struct _WebKitWebViewAccessible WebKitWebViewAccessible;
typedef struct _WebKitWebViewAccessibleClass WebKitWebViewAccessibleClass;
typedef struct _WebKitWebViewAccessiblePrivate WebKitWebViewAccessiblePrivate;

struct _WebKitWebViewAccessible {
    AtkSocket parent;
    /*< private >*/
    WebKitWebViewAccessiblePrivate* priv;
};

struct _WebKitWebViewAccessibleClass {
    AtkSocketClass parentClass;
};

GType webkit_web_view_accessible_get_type();

WebKitWebViewAccessible* webkitWebViewAccessibleNew(gpointer);
void webkitWebViewAccessibleSetWebView(WebKitWebViewAccessible*, gpointer);

G_END_DECLS

#endif // USE(ATK)
