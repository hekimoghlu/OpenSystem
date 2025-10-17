/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 31, 2022.
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
#include "config.h"
#include "WebKitPermissionRequest.h"

#if !ENABLE(2022_GLIB_API)
typedef WebKitPermissionRequestIface WebKitPermissionRequestInterface;
#endif

/**
 * WebKitPermissionRequest:
 * @See_also: #WebKitWebView
 *
 * A permission request.
 *
 * There are situations where an embedder would need to ask the user
 * for permission to do certain types of operations, such as switching
 * to fullscreen mode or reporting the user's location through the
 * standard Geolocation API. In those cases, WebKit will emit a
 * #WebKitWebView::permission-request signal with a
 * #WebKitPermissionRequest object attached to it.
 */

G_DEFINE_INTERFACE(WebKitPermissionRequest, webkit_permission_request, G_TYPE_OBJECT)

static void webkit_permission_request_default_init(WebKitPermissionRequestInterface*)
{
}

/**
 * webkit_permission_request_allow:
 * @request: a #WebKitPermissionRequest
 *
 * Allow the action which triggered this request.
 */
void webkit_permission_request_allow(WebKitPermissionRequest* request)
{
    g_return_if_fail(WEBKIT_IS_PERMISSION_REQUEST(request));

    WebKitPermissionRequestInterface* iface = WEBKIT_PERMISSION_REQUEST_GET_IFACE(request);
    if (iface->allow)
        iface->allow(request);
}

/**
 * webkit_permission_request_deny:
 * @request: a #WebKitPermissionRequest
 *
 * Deny the action which triggered this request.
 */
void webkit_permission_request_deny(WebKitPermissionRequest* request)
{
    g_return_if_fail(WEBKIT_IS_PERMISSION_REQUEST(request));

    WebKitPermissionRequestInterface* iface = WEBKIT_PERMISSION_REQUEST_GET_IFACE(request);
    if (iface->deny)
        iface->deny(request);
}
