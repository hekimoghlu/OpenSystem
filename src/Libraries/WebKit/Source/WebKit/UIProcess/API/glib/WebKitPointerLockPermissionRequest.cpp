/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 10, 2025.
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
#include "WebKitPointerLockPermissionRequest.h"

#include "WebKitPermissionRequest.h"
#include "WebKitPointerLockPermissionRequestPrivate.h"
#include "WebKitWebViewPrivate.h"
#include <wtf/glib/WTFGType.h>

/**
 * WebKitPointerLockPermissionRequest:
 * @See_also: #WebKitPermissionRequest, #WebKitWebView
 *
 * A permission request for locking the pointer.
 *
 * WebKitPointerLockPermissionRequest represents a request for
 * permission to decide whether WebKit can lock the pointer device when
 * requested by web content.
 *
 * When a WebKitPointerLockPermissionRequest is not handled by the user,
 * it is allowed by default.
 *
 * Since: 2.28
 */

#if !ENABLE(2022_GLIB_API)
typedef WebKitPermissionRequestIface WebKitPermissionRequestInterface;
#endif

static void webkit_permission_request_interface_init(WebKitPermissionRequestInterface*);

struct _WebKitPointerLockPermissionRequestPrivate {
    GRefPtr<WebKitWebView> webView;
    bool madeDecision;
};

WEBKIT_DEFINE_FINAL_TYPE_WITH_CODE(
    WebKitPointerLockPermissionRequest, webkit_pointer_lock_permission_request, G_TYPE_OBJECT, GObject,
    G_IMPLEMENT_INTERFACE(WEBKIT_TYPE_PERMISSION_REQUEST, webkit_permission_request_interface_init))

#if ENABLE(POINTER_LOCK)
static void webkitPointerLockPermissionRequestAllow(WebKitPermissionRequest* request)
{
    ASSERT(WEBKIT_IS_POINTER_LOCK_PERMISSION_REQUEST(request));

    WebKitPointerLockPermissionRequestPrivate* priv = WEBKIT_POINTER_LOCK_PERMISSION_REQUEST(request)->priv;

    // Only one decision at a time.
    if (priv->madeDecision)
        return;

    webkitWebViewRequestPointerLock(priv->webView.get());
    priv->madeDecision = true;
}

static void webkitPointerLockPermissionRequestDeny(WebKitPermissionRequest* request)
{
    ASSERT(WEBKIT_IS_POINTER_LOCK_PERMISSION_REQUEST(request));

    WebKitPointerLockPermissionRequestPrivate* priv = WEBKIT_POINTER_LOCK_PERMISSION_REQUEST(request)->priv;

    // Only one decision at a time.
    if (priv->madeDecision)
        return;

    webkitWebViewDenyPointerLockRequest(priv->webView.get());
    priv->madeDecision = true;
}
#endif // ENABLE(POINTER_LOCK)

static void webkit_permission_request_interface_init(WebKitPermissionRequestInterface* iface)
{
#if ENABLE(POINTER_LOCK)
    iface->allow = webkitPointerLockPermissionRequestAllow;
    iface->deny = webkitPointerLockPermissionRequestDeny;
#else
    iface->allow = nullptr;
    iface->deny = nullptr;
#endif // ENABLE(POINTER_LOCK)
}

static void webkitPointerLockPermissionRequestDispose(GObject* object)
{
    // Default behaviour when no decision has been made is allowing the request.
#if ENABLE(POINTER_LOCK)
    webkitPointerLockPermissionRequestAllow(WEBKIT_PERMISSION_REQUEST(object));
#endif // ENABLE(POINTER_LOCK)
    G_OBJECT_CLASS(webkit_pointer_lock_permission_request_parent_class)->dispose(object);
}

static void webkit_pointer_lock_permission_request_class_init(WebKitPointerLockPermissionRequestClass* klass)
{
    GObjectClass* objectClass = G_OBJECT_CLASS(klass);
    objectClass->dispose = webkitPointerLockPermissionRequestDispose;
}

WebKitPointerLockPermissionRequest* webkitPointerLockPermissionRequestCreate(WebKitWebView* webView)
{
    WebKitPointerLockPermissionRequest* request = WEBKIT_POINTER_LOCK_PERMISSION_REQUEST(g_object_new(WEBKIT_TYPE_POINTER_LOCK_PERMISSION_REQUEST, nullptr));
    request->priv->webView = webView;
    return request;
}

void webkitPointerLockPermissionRequestDidLosePointerLock(WebKitPointerLockPermissionRequest* request)
{
    request->priv->madeDecision = true;
    request->priv->webView = nullptr;
}
