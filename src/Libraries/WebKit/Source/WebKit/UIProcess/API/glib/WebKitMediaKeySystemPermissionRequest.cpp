/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 5, 2022.
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
#include "WebKitMediaKeySystemPermissionRequest.h"

#include "MediaKeySystemPermissionRequest.h"
#include "WebKitMediaKeySystemPermissionRequestPrivate.h"
#include "WebKitPermissionRequest.h"
#include <wtf/glib/WTFGType.h>

#if !ENABLE(2022_GLIB_API)
typedef WebKitPermissionRequestIface WebKitPermissionRequestInterface;
#endif

using namespace WebKit;

/**
 * WebKitMediaKeySystemPermissionRequest:
 * @See_also: #WebKitPermissionRequest, #WebKitWebView
 *
 * A permission request for using an EME Content Decryption Module.
 *
 * WebKitMediaKeySystemPermissionRequest represents a request for permission to decide whether
 * WebKit should use the given CDM to access protected media when requested through the
 * MediaKeySystem API.
 *
 * When a WebKitMediaKeySystemPermissionRequest is not handled by the user,
 * it is denied by default.
 *
 * When handling this permission request the application may perform additional installation of the
 * requested CDM, unless it is already present on the host system.
 */

static void webkit_permission_request_interface_init(WebKitPermissionRequestInterface*);

struct _WebKitMediaKeySystemPermissionRequestPrivate {
    RefPtr<MediaKeySystemPermissionRequest> request;
    bool madeDecision;
    CString keySystem;
};

WEBKIT_DEFINE_FINAL_TYPE_WITH_CODE(
    WebKitMediaKeySystemPermissionRequest, webkit_media_key_system_permission_request, G_TYPE_OBJECT, GObject,
    G_IMPLEMENT_INTERFACE(WEBKIT_TYPE_PERMISSION_REQUEST, webkit_permission_request_interface_init))

static void webkitMediaKeySystemPermissionRequestAllow(WebKitPermissionRequest* request)
{
    ASSERT(WEBKIT_IS_MEDIA_KEY_SYSTEM_PERMISSION_REQUEST(request));

    WebKitMediaKeySystemPermissionRequestPrivate* priv = WEBKIT_MEDIA_KEY_SYSTEM_PERMISSION_REQUEST(request)->priv;

    // Only one decision at a time.
    if (priv->madeDecision)
        return;

    priv->request->complete(true);
    priv->madeDecision = true;
}

static void webkitMediaKeySystemPermissionRequestDeny(WebKitPermissionRequest* request)
{
    ASSERT(WEBKIT_IS_MEDIA_KEY_SYSTEM_PERMISSION_REQUEST(request));

    WebKitMediaKeySystemPermissionRequestPrivate* priv = WEBKIT_MEDIA_KEY_SYSTEM_PERMISSION_REQUEST(request)->priv;

    // Only one decision at a time.
    if (priv->madeDecision)
        return;

    priv->request->complete(false);
    priv->madeDecision = true;
}

static void webkit_permission_request_interface_init(WebKitPermissionRequestInterface* iface)
{
    iface->allow = webkitMediaKeySystemPermissionRequestAllow;
    iface->deny = webkitMediaKeySystemPermissionRequestDeny;
}

static void webkitMediaKeySystemPermissionRequestDispose(GObject* object)
{
    // Default behaviour when no decision has been made is denying the request.
    webkitMediaKeySystemPermissionRequestDeny(WEBKIT_PERMISSION_REQUEST(object));
    G_OBJECT_CLASS(webkit_media_key_system_permission_request_parent_class)->dispose(object);
}

static void webkit_media_key_system_permission_request_class_init(WebKitMediaKeySystemPermissionRequestClass* klass)
{
    GObjectClass* objectClass = G_OBJECT_CLASS(klass);
    objectClass->dispose = webkitMediaKeySystemPermissionRequestDispose;
}

WebKitMediaKeySystemPermissionRequest* webkitMediaKeySystemPermissionRequestCreate(Ref<MediaKeySystemPermissionRequest>&& request)
{
    WebKitMediaKeySystemPermissionRequest* permissionRequest = WEBKIT_MEDIA_KEY_SYSTEM_PERMISSION_REQUEST(g_object_new(WEBKIT_TYPE_MEDIA_KEY_SYSTEM_PERMISSION_REQUEST, NULL));
    permissionRequest->priv->request = WTFMove(request);
    return permissionRequest;
}

/**
 * webkit_media_key_system_permission_get_name:
 * @request: a #WebKitMediaKeySystemPermissionRequest
 *
 * Get the key system for which access permission is being requested.
 *
 * Returns: the key system name for @request
 *
 * Since: 2.32
 */
const gchar*
webkit_media_key_system_permission_get_name(WebKitMediaKeySystemPermissionRequest* request)
{
    auto* priv = request->priv;
    if (priv->keySystem.isNull())
        priv->keySystem = priv->request->keySystem().utf8().data();
    return priv->keySystem.data();
}
