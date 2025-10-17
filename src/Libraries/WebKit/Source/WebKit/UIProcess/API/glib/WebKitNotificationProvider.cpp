/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 13, 2023.
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
#include "WebKitNotificationProvider.h"

#include "APIArray.h"
#include "APINotificationProvider.h"
#include "APINumber.h"
#include "NotificationService.h"
#include "WebKitNotificationPrivate.h"
#include "WebKitWebContextPrivate.h"
#include "WebKitWebViewPrivate.h"
#include "WebNotificationManagerProxy.h"
#include "WebPageProxy.h"
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/CString.h>

using namespace WebKit;


class NotificationProvider final : public API::NotificationProvider {
public:
    explicit NotificationProvider(WebKitNotificationProvider& provider)
        : m_provider(provider)
    {
    }

private:
    bool show(WebPageProxy* page, WebNotification& notification, RefPtr<WebCore::NotificationResources>&& resources) override
    {
        m_provider.show(page, notification, WTFMove(resources));
        return true;
    }

    void cancel(WebNotification& notification) override
    {
        m_provider.cancel(notification);
    }

    void clearNotifications(const Vector<WebNotificationIdentifier>& notificationIDs) override
    {
        m_provider.clearNotifications(notificationIDs);
    }

    HashMap<String, bool> notificationPermissions() override
    {
        return m_provider.notificationPermissions();
    }

    WebKitNotificationProvider& m_provider;
};

WTF_MAKE_TZONE_ALLOCATED_IMPL(WebKitNotificationProvider);

WebKitNotificationProvider::WebKitNotificationProvider(WebNotificationManagerProxy* notificationManager, WebKitWebContext* webContext)
    : m_webContext(webContext)
    , m_notificationManager(notificationManager)
{
    ASSERT(m_notificationManager);
    m_notificationManager->setProvider(makeUnique<NotificationProvider>(*this));
}

WebKitNotificationProvider::~WebKitNotificationProvider()
{
    if (m_observerRegistered)
        NotificationService::singleton().removeObserver(*this);

    m_notificationManager->setProvider(nullptr);
}

void WebKitNotificationProvider::apiNotificationCloseCallback(WebKitNotification* notification, WebKitNotificationProvider* provider)
{
    uint64_t notificationID = webkit_notification_get_id(notification);
    Vector<RefPtr<API::Object>> arrayIDs;
    arrayIDs.append(API::UInt64::create(notificationID));
    provider->m_notificationManager->providerDidCloseNotifications(API::Array::create(WTFMove(arrayIDs)).ptr());
    provider->m_apiNotifications.remove(WebNotificationIdentifier { notificationID });
}

void WebKitNotificationProvider::apiNotificationClickedCallback(WebKitNotification* notification, WebKitNotificationProvider* provider)
{
    provider->m_notificationManager->providerDidClickNotification(WebNotificationIdentifier { webkit_notification_get_id(notification) });
}

void WebKitNotificationProvider::closeAPINotification(WebNotificationIdentifier notificationID)
{
    if (auto notification = m_apiNotifications.take(notificationID))
        webkit_notification_close(notification.get());
}

void WebKitNotificationProvider::withdrawAnyPreviousAPINotificationMatchingTag(const CString& tag)
{
    if (!tag.length())
        return;

    for (auto& notification : m_apiNotifications.values()) {
        if (tag == webkit_notification_get_tag(notification.get())) {
            closeAPINotification(WebNotificationIdentifier { webkit_notification_get_id(notification.get()) });
            break;
        }
    }

#if ASSERT_ENABLED
    for (auto& notification : m_apiNotifications.values())
        ASSERT(tag != webkit_notification_get_tag(notification.get()));
#endif
}

void WebKitNotificationProvider::show(WebPageProxy* page, WebNotification& webNotification, RefPtr<WebCore::NotificationResources>&& resources)
{
    if (!page || !m_webContext) {
        // FIXME: glib API needs to find their own solution to handling pageless notifications.
        show(webNotification, resources);
        return;
    }

    GRefPtr<WebKitNotification> notification = m_apiNotifications.get(webNotification.identifier());
    if (!notification) {
        withdrawAnyPreviousAPINotificationMatchingTag(webNotification.tag().utf8());
        notification = adoptGRef(webkitNotificationCreate(webNotification));
        g_signal_connect(notification.get(), "closed", G_CALLBACK(apiNotificationCloseCallback), this);
        g_signal_connect(notification.get(), "clicked", G_CALLBACK(apiNotificationClickedCallback), this);
        m_apiNotifications.set(webNotification.identifier(), notification);
    }

    auto* webView = webkitWebContextGetWebViewForPage(m_webContext, page);
    ASSERT(webView);

    if (webkitWebViewEmitShowNotification(webView, notification.get()))
        m_notificationManager->providerDidShowNotification(webNotification.identifier());
    else {
        g_signal_handlers_disconnect_by_data(notification.get(), this);
        show(webNotification, resources);
    }
}

void WebKitNotificationProvider::show(WebNotification& webNotification, const RefPtr<WebCore::NotificationResources>& resources)
{
    if (!m_observerRegistered) {
        NotificationService::singleton().addObserver(*this);
        m_observerRegistered = true;
    }

    if (NotificationService::singleton().showNotification(webNotification, resources))
        m_notificationManager->providerDidShowNotification(webNotification.identifier());
}

void WebKitNotificationProvider::cancelNotificationByID(WebNotificationIdentifier notificationID)
{
    closeAPINotification(notificationID);

    if (m_observerRegistered)
        NotificationService::singleton().cancelNotification(notificationID);
}

void WebKitNotificationProvider::cancel(const WebNotification& webNotification)
{
    cancelNotificationByID(webNotification.identifier());
}

void WebKitNotificationProvider::clearNotifications(const Vector<WebNotificationIdentifier>& notificationIDs)
{
    for (const auto& item : notificationIDs)
        cancelNotificationByID(item);
}

HashMap<WTF::String, bool> WebKitNotificationProvider::notificationPermissions()
{
    if (m_webContext)
        webkitWebContextInitializeNotificationPermissions(m_webContext);
    return m_notificationPermissions;
}

void WebKitNotificationProvider::setNotificationPermissions(HashMap<String, bool>&& permissionsMap)
{
    m_notificationPermissions = WTFMove(permissionsMap);
}

void WebKitNotificationProvider::didClickNotification(WebNotificationIdentifier notificationID)
{
    if (auto* notification = m_apiNotifications.get(notificationID))
        webkit_notification_clicked(notification);

    m_notificationManager->providerDidClickNotification(notificationID);
}

void WebKitNotificationProvider::didCloseNotification(WebNotificationIdentifier notificationID)
{
    closeAPINotification(notificationID);

    Vector<RefPtr<API::Object>> arrayIDs;
    arrayIDs.append(API::UInt64::create(notificationID.toUInt64()));
    m_notificationManager->providerDidCloseNotifications(API::Array::create(WTFMove(arrayIDs)).ptr());
}
