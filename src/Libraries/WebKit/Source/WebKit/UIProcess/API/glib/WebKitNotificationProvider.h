/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 26, 2023.
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

#include "NotificationService.h"
#include "WebKitNotification.h"
#include "WebKitWebContext.h"
#include <wtf/Forward.h>
#include <wtf/HashMap.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/glib/GRefPtr.h>
#include <wtf/text/StringHash.h>

namespace WebCore {
class NotificationResources;
}

namespace WebKit {
class WebNotificationManagerProxy;
class WebNotification;
class WebPageProxy;

class WebKitNotificationProvider final : public NotificationService::Observer {
    WTF_MAKE_TZONE_ALLOCATED(WebKitNotificationProvider);
public:
    WebKitNotificationProvider(WebNotificationManagerProxy*, WebKitWebContext*);
    ~WebKitNotificationProvider();

    void show(WebPageProxy*, WebNotification&, RefPtr<WebCore::NotificationResources>&&);
    void cancel(const WebNotification&);
    void clearNotifications(const Vector<WebNotificationIdentifier>&);

    HashMap<WTF::String, bool> notificationPermissions();
    void setNotificationPermissions(HashMap<String, bool>&&);

private:
    void cancelNotificationByID(WebNotificationIdentifier);
    static void apiNotificationCloseCallback(WebKitNotification*, WebKitNotificationProvider*);
    static void apiNotificationClickedCallback(WebKitNotification*, WebKitNotificationProvider*);
    static void apiNotificationWeakNotify(gpointer, GObject*);
    void addAPINotification(WebKitNotification*);
    void removeAPINotification(WebKitNotification*);
    void removeAPINotification(WebNotificationIdentifier);
    void closeAPINotification(WebNotificationIdentifier);

    void withdrawAnyPreviousAPINotificationMatchingTag(const CString&);

    void show(WebNotification&, const RefPtr<WebCore::NotificationResources>&);

    // NotificationService
    void didClickNotification(WebNotificationIdentifier) final;
    void didCloseNotification(WebNotificationIdentifier) final;

    WebKitWebContext* m_webContext;
    HashMap<WTF::String, bool> m_notificationPermissions;
    RefPtr<WebNotificationManagerProxy> m_notificationManager;
    HashMap<WebNotificationIdentifier, GRefPtr<WebKitNotification>> m_apiNotifications;
    bool m_observerRegistered { false };
};

} // namespace WebKit
