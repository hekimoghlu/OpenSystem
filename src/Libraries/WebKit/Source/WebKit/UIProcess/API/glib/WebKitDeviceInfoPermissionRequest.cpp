/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 13, 2022.
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
#include "WebKitDeviceInfoPermissionRequest.h"

#include "DeviceIdHashSaltStorage.h"
#include "UserMediaPermissionCheckProxy.h"
#include "WebKitDeviceInfoPermissionRequestPrivate.h"
#include "WebKitPermissionRequest.h"
#include "WebsiteDataStore.h"
#include <glib/gi18n-lib.h>
#include <wtf/glib/WTFGType.h>

#if !ENABLE(2022_GLIB_API)
typedef WebKitPermissionRequestIface WebKitPermissionRequestInterface;
#endif

using namespace WebKit;

/**
 * WebKitDeviceInfoPermissionRequest:
 * @See_also: #WebKitPermissionRequest, #WebKitWebView
 *
 * A permission request for accessing user's audio/video devices.
 *
 * WebKitUserMediaPermissionRequest represents a request for
 * permission to whether WebKit should be allowed to access the user's
 * devices information when requested through the enumerateDevices API.
 *
 * When a WebKitDeviceInfoPermissionRequest is not handled by the user,
 * it is denied by default.
 *
 * Since: 2.24
 */

static void webkit_permission_request_interface_init(WebKitPermissionRequestInterface*);

struct _WebKitDeviceInfoPermissionRequestPrivate {
    RefPtr<UserMediaPermissionCheckProxy> request;
    RefPtr<DeviceIdHashSaltStorage> deviceIdHashSaltStorage;
    bool madeDecision;
};

WEBKIT_DEFINE_FINAL_TYPE_WITH_CODE(
    WebKitDeviceInfoPermissionRequest, webkit_device_info_permission_request, G_TYPE_OBJECT, GObject,
    G_IMPLEMENT_INTERFACE(WEBKIT_TYPE_PERMISSION_REQUEST, webkit_permission_request_interface_init))

static void webkitDeviceInfoPermissionRequestAllow(WebKitPermissionRequest* request)
{
    ASSERT(WEBKIT_IS_DEVICE_INFO_PERMISSION_REQUEST(request));

    auto* priv = WEBKIT_DEVICE_INFO_PERMISSION_REQUEST(request)->priv;

    if (!priv->deviceIdHashSaltStorage) {
        priv->request->setUserMediaAccessInfo(false);
        return;
    }

    // Only one decision at a time.
    if (priv->madeDecision)
        return;

    priv->madeDecision = true;
    priv->request->setUserMediaAccessInfo(true);
}

static void webkitDeviceInfoPermissionRequestDeny(WebKitPermissionRequest* request)
{
    ASSERT(WEBKIT_IS_DEVICE_INFO_PERMISSION_REQUEST(request));

    auto* priv = WEBKIT_DEVICE_INFO_PERMISSION_REQUEST(request)->priv;

    if (!priv->deviceIdHashSaltStorage) {
        priv->request->setUserMediaAccessInfo(false);
        return;
    }

    // Only one decision at a time.
    if (priv->madeDecision)
        return;

    priv->madeDecision = true;
    priv->request->setUserMediaAccessInfo(false);
}

static void webkit_permission_request_interface_init(WebKitPermissionRequestInterface* iface)
{
    iface->allow = webkitDeviceInfoPermissionRequestAllow;
    iface->deny = webkitDeviceInfoPermissionRequestDeny;
}

static void webkitDeviceInfoPermissionRequestDispose(GObject* object)
{
    // Default behaviour when no decision has been made is denying the request.
    webkitDeviceInfoPermissionRequestDeny(WEBKIT_PERMISSION_REQUEST(object));
    G_OBJECT_CLASS(webkit_device_info_permission_request_parent_class)->dispose(object);
}

static void webkit_device_info_permission_request_class_init(WebKitDeviceInfoPermissionRequestClass* klass)
{
    GObjectClass* objectClass = G_OBJECT_CLASS(klass);
    objectClass->dispose = webkitDeviceInfoPermissionRequestDispose;
}

WebKitDeviceInfoPermissionRequest* webkitDeviceInfoPermissionRequestCreate(UserMediaPermissionCheckProxy& request, DeviceIdHashSaltStorage* deviceIdHashSaltStorage)
{
    auto* deviceInfoPermissionRequest = WEBKIT_DEVICE_INFO_PERMISSION_REQUEST(g_object_new(WEBKIT_TYPE_DEVICE_INFO_PERMISSION_REQUEST, nullptr));

    deviceInfoPermissionRequest->priv->request = &request;
    deviceInfoPermissionRequest->priv->deviceIdHashSaltStorage = deviceIdHashSaltStorage;
    return deviceInfoPermissionRequest;
}
