/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 22, 2025.
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
#include "WebKitNotificationPermissionRequest.h"

#include "NotificationPermissionRequest.h"
#include "WebKitNotificationPermissionRequestPrivate.h"
#include "WebKitPermissionRequest.h"
#include <wtf/glib/WTFGType.h>

#if !ENABLE(2022_GLIB_API)
typedef WebKitPermissionRequestIface WebKitPermissionRequestInterface;
#endif

using namespace WebKit;

/**
 * WebKitNotificationPermissionRequest:
 * @See_also: #WebKitPermissionRequest, #WebKitWebView
 *
 * A permission request for displaying web notifications.
 *
 * WebKitNotificationPermissionRequest represents a request for
 * permission to decide whether WebKit should provide the user with
 * notifications through the Web Notification API.
 *
 * When a WebKitNotificationPermissionRequest is not handled by the user,
 * it is denied by default.
 *
 * Since: 2.8
 */

static void webkit_permission_request_interface_init(WebKitPermissionRequestInterface*);

struct _WebKitNotificationPermissionRequestPrivate {
    RefPtr<NotificationPermissionRequest> request;
    bool madeDecision;
};

WEBKIT_DEFINE_FINAL_TYPE_WITH_CODE(
    WebKitNotificationPermissionRequest, webkit_notification_permission_request, G_TYPE_OBJECT, GObject,
    G_IMPLEMENT_INTERFACE(WEBKIT_TYPE_PERMISSION_REQUEST, webkit_permission_request_interface_init))

static void webkitNotificationPermissionRequestAllow(WebKitPermissionRequest* request)
{
    ASSERT(WEBKIT_IS_NOTIFICATION_PERMISSION_REQUEST(request));

    WebKitNotificationPermissionRequestPrivate* priv = WEBKIT_NOTIFICATION_PERMISSION_REQUEST(request)->priv;

    // Only one decision at a time.
    if (priv->madeDecision)
        return;

    priv->request->didReceiveDecision(true);
    priv->madeDecision = true;
}

static void webkitNotificationPermissionRequestDeny(WebKitPermissionRequest* request)
{
    ASSERT(WEBKIT_IS_NOTIFICATION_PERMISSION_REQUEST(request));

    WebKitNotificationPermissionRequestPrivate* priv = WEBKIT_NOTIFICATION_PERMISSION_REQUEST(request)->priv;

    // Only one decision at a time.
    if (priv->madeDecision)
        return;

    priv->request->didReceiveDecision(false);
    priv->madeDecision = true;
}

static void webkit_permission_request_interface_init(WebKitPermissionRequestInterface* iface)
{
    iface->allow = webkitNotificationPermissionRequestAllow;
    iface->deny = webkitNotificationPermissionRequestDeny;
}

static void webkitNotificationPermissionRequestDispose(GObject* object)
{
    // Default behaviour when no decision has been made is denying the request.
    webkitNotificationPermissionRequestDeny(WEBKIT_PERMISSION_REQUEST(object));
    G_OBJECT_CLASS(webkit_notification_permission_request_parent_class)->dispose(object);
}

static void webkit_notification_permission_request_class_init(WebKitNotificationPermissionRequestClass* klass)
{
    GObjectClass* objectClass = G_OBJECT_CLASS(klass);
    objectClass->dispose = webkitNotificationPermissionRequestDispose;
}

WebKitNotificationPermissionRequest* webkitNotificationPermissionRequestCreate(NotificationPermissionRequest* request)
{
    WebKitNotificationPermissionRequest* notificationPermissionRequest = WEBKIT_NOTIFICATION_PERMISSION_REQUEST(g_object_new(WEBKIT_TYPE_NOTIFICATION_PERMISSION_REQUEST, nullptr));
    notificationPermissionRequest->priv->request = request;
    return notificationPermissionRequest;
}
