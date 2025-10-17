/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 10, 2022.
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
#include "WebKitClipboardPermissionRequest.h"

#include "WebKitClipboardPermissionRequestPrivate.h"
#include "WebKitPermissionRequest.h"
#include <wtf/glib/WTFGType.h>

#if !ENABLE(2022_GLIB_API)
typedef WebKitPermissionRequestIface WebKitPermissionRequestInterface;
#endif

/**
 * WebKitClipboardPermissionRequest:
 * @See_also: #WebKitPermissionRequest, #WebKitWebView
 *
 * A permission request for reading clipboard contents.
 *
 * WebKitClipboardPermissionRequest represents a request for
 * permission to decide whether WebKit can access the clipboard to read
 * its contents through the Async Clipboard API.
 *
 * When a WebKitClipboardPermissionRequest is not handled by the user,
 * it is denied by default.
 *
 * Since: 2.42
 */

static void webkit_permission_request_interface_init(WebKitPermissionRequestInterface*);

struct _WebKitClipboardPermissionRequestPrivate {
    CompletionHandler<void(WebCore::DOMPasteAccessResponse)> completionHandler;
};

WEBKIT_DEFINE_FINAL_TYPE_WITH_CODE(
    WebKitClipboardPermissionRequest, webkit_clipboard_permission_request, G_TYPE_OBJECT, GObject,
    G_IMPLEMENT_INTERFACE(WEBKIT_TYPE_PERMISSION_REQUEST, webkit_permission_request_interface_init))

static void webkitClipboardPermissionRequestAllow(WebKitPermissionRequest* request)
{
    ASSERT(WEBKIT_IS_CLIPBOARD_PERMISSION_REQUEST(request));

    WebKitClipboardPermissionRequestPrivate* priv = WEBKIT_CLIPBOARD_PERMISSION_REQUEST(request)->priv;

    if (priv->completionHandler)
        priv->completionHandler(WebCore::DOMPasteAccessResponse::GrantedForGesture);
}

static void webkitClipboardPermissionRequestDeny(WebKitPermissionRequest* request)
{
    ASSERT(WEBKIT_IS_CLIPBOARD_PERMISSION_REQUEST(request));

    WebKitClipboardPermissionRequestPrivate* priv = WEBKIT_CLIPBOARD_PERMISSION_REQUEST(request)->priv;

    if (priv->completionHandler)
        priv->completionHandler(WebCore::DOMPasteAccessResponse::DeniedForGesture);
}

static void webkit_permission_request_interface_init(WebKitPermissionRequestInterface* iface)
{
    iface->allow = webkitClipboardPermissionRequestAllow;
    iface->deny = webkitClipboardPermissionRequestDeny;
}

static void webkitClipboardPermissionRequestDispose(GObject* object)
{
    // Default behaviour when no decision has been made is denying the request.
    webkitClipboardPermissionRequestDeny(WEBKIT_PERMISSION_REQUEST(object));
    G_OBJECT_CLASS(webkit_clipboard_permission_request_parent_class)->dispose(object);
}

static void webkit_clipboard_permission_request_class_init(WebKitClipboardPermissionRequestClass* klass)
{
    GObjectClass* objectClass = G_OBJECT_CLASS(klass);
    objectClass->dispose = webkitClipboardPermissionRequestDispose;
}

WebKitClipboardPermissionRequest* webkitClipboardPermissionRequestCreate(CompletionHandler<void(WebCore::DOMPasteAccessResponse)>&& completionHandler)
{
    WebKitClipboardPermissionRequest* clipboardPermissionRequest = WEBKIT_CLIPBOARD_PERMISSION_REQUEST(g_object_new(WEBKIT_TYPE_CLIPBOARD_PERMISSION_REQUEST, nullptr));
    clipboardPermissionRequest->priv->completionHandler = WTFMove(completionHandler);
    return clipboardPermissionRequest;
}
