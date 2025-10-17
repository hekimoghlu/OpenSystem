/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 29, 2021.
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

#include "APIObject.h"
#include "MessageReceiver.h"
#include "WebContextSupplement.h"
#include "WebPageProxyIdentifier.h"
#include <WebCore/NotificationClient.h>
#include <pal/SessionID.h>
#include <wtf/HashMap.h>
#include <wtf/UUID.h>
#include <wtf/text/StringHash.h>

namespace WebCore {
class NotificationResources;
struct NotificationData;
}

namespace API {
class Array;
class NotificationProvider;
class SecurityOrigin;
}

namespace WebKit {

class WebNotification;
class WebPageProxy;
class WebProcessPool;
class WebsiteDataStore;

enum class WebNotificationIdentifierType;
using WebNotificationIdentifier = ObjectIdentifier<WebNotificationIdentifierType>;

class WebNotificationManagerProxy : public API::ObjectImpl<API::Object::Type::NotificationManager>, public WebContextSupplement {
public:
    static ASCIILiteral supplementName();

    static Ref<WebNotificationManagerProxy> create(WebProcessPool*);

    static WebNotificationManagerProxy& sharedServiceWorkerManager();
    static Ref<WebNotificationManagerProxy> protectedSharedServiceWorkerManager();

    virtual ~WebNotificationManagerProxy();

    void setProvider(std::unique_ptr<API::NotificationProvider>&&);
    HashMap<String, bool> notificationPermissions();

    void show(WebPageProxy*, IPC::Connection&, const WebCore::NotificationData&, RefPtr<WebCore::NotificationResources>&&);
    bool showPersistent(const WebsiteDataStore&, IPC::Connection*, const WebCore::NotificationData&, RefPtr<WebCore::NotificationResources>&&);
    void cancel(WebPageProxy*, const WTF::UUID& pageNotificationID);
    void clearNotifications(WebPageProxy*);
    void clearNotifications(WebPageProxy*, const Vector<WTF::UUID>& pageNotificationIDs);
    void didDestroyNotification(WebPageProxy*, const WTF::UUID& pageNotificationID);

    void getNotifications(const URL&, const String&, PAL::SessionID, CompletionHandler<void(Vector<WebCore::NotificationData>&&)>&&);

    void providerDidShowNotification(WebNotificationIdentifier);
    void providerDidClickNotification(WebNotificationIdentifier);
    void providerDidClickNotification(const WTF::UUID& notificationID);
    void providerDidCloseNotifications(API::Array* notificationIDs);
    void providerDidUpdateNotificationPolicy(const API::SecurityOrigin*, bool allowed);
    void providerDidRemoveNotificationPolicies(API::Array* origins);

    using API::Object::ref;
    using API::Object::deref;

private:
    explicit WebNotificationManagerProxy(WebProcessPool*);

    // WebContextSupplement
    void processPoolDestroyed() override;
    void refWebContextSupplement() override;
    void derefWebContextSupplement() override;

    bool showImpl(WebPageProxy*, Ref<WebNotification>&&, RefPtr<WebCore::NotificationResources>&&);

    std::unique_ptr<API::NotificationProvider> m_provider;

    HashMap<WebNotificationIdentifier, WTF::UUID> m_globalNotificationMap;
    HashMap<WTF::UUID, Ref<WebNotification>> m_notifications;
};

} // namespace WebKit
