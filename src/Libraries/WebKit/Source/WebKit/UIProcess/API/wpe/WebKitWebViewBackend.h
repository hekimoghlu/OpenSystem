/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 21, 2024.
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
#if !defined(__WEBKIT_H_INSIDE__) && !defined(BUILDING_WEBKIT)
#error "Only <wpe/webkit.h> can be included directly."
#endif

#ifndef WebKitWebViewBackend_h
#define WebKitWebViewBackend_h

#include <glib-object.h>
#include <wpe/WebKitDefines.h>
#include <wpe/wpe.h>

G_BEGIN_DECLS

#define WEBKIT_TYPE_WEB_VIEW_BACKEND (webkit_web_view_backend_get_type())

typedef struct _WebKitWebViewBackend WebKitWebViewBackend;

WEBKIT_API GType
webkit_web_view_backend_get_type        (void);

WEBKIT_API WebKitWebViewBackend *
webkit_web_view_backend_new             (struct wpe_view_backend *backend,
                                         GDestroyNotify           notify,
                                         gpointer                 user_data);
WEBKIT_API struct wpe_view_backend *
webkit_web_view_backend_get_wpe_backend (WebKitWebViewBackend    *view_backend);

G_END_DECLS

#endif /* WebKitWebViewBackend_h */
