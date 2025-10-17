/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 17, 2024.
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

#include "NotificationManagerMessageHandler.h"

namespace WebKit {

class WebPageProxy;

class WebNotificationManagerMessageHandler : public NotificationManagerMessageHandler {
    friend class WebPageProxy;

public:
    void ref() const final;
    void deref() const final;

private:
    explicit WebNotificationManagerMessageHandler(WebPageProxy&);

    void showNotification(IPC::Connection&, const WebCore::NotificationData&, RefPtr<WebCore::NotificationResources>&&, CompletionHandler<void()>&&) final;
    void cancelNotification(WebCore::SecurityOriginData&&, const WTF::UUID& notificationID) final;
    void clearNotifications(const Vector<WTF::UUID>& notificationIDs) final;
    void didDestroyNotification(const WTF::UUID& notificationID) final;
    void pageWasNotifiedOfNotificationPermission() final;
    void requestPermission(WebCore::SecurityOriginData&&, CompletionHandler<void(bool)>&&) final;
    void setAppBadge(const WebCore::SecurityOriginData&, std::optional<uint64_t>) final { }
    void getPermissionState(WebCore::SecurityOriginData&&, CompletionHandler<void(WebCore::PushPermissionState)>&&) final;
    void getPermissionStateSync(WebCore::SecurityOriginData&&, CompletionHandler<void(WebCore::PushPermissionState)>&&) final;
    std::optional<SharedPreferencesForWebProcess> sharedPreferencesForWebProcess(const IPC::Connection&) const final;
    Ref<WebPageProxy> protectedPage() const;

    WeakRef<WebPageProxy> m_webPageProxy;
};

} // namespace WebKit
