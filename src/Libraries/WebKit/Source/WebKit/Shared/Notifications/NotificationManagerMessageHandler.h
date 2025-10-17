/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 10, 2022.
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

#include "MessageReceiver.h"
#include "SharedPreferencesForWebProcess.h"
#include "WebPageProxyIdentifier.h"
#include <WebCore/NotificationDirection.h>
#include <wtf/AbstractRefCounted.h>
#include <wtf/UUID.h>

namespace WebCore {
enum class PushPermissionState : uint8_t;
class NotificationResources;
class SecurityOriginData;
struct NotificationData;
}

namespace WebKit {

class NotificationManagerMessageHandler : public IPC::MessageReceiver {
public:
    virtual ~NotificationManagerMessageHandler() = default;

    virtual void showNotification(IPC::Connection&, const WebCore::NotificationData&, RefPtr<WebCore::NotificationResources>&&, CompletionHandler<void()>&&) = 0;
    virtual void cancelNotification(WebCore::SecurityOriginData&&, const WTF::UUID& notificationID) = 0;
    virtual void clearNotifications(const Vector<WTF::UUID>& notificationIDs) = 0;
    virtual void didDestroyNotification(const WTF::UUID& notificationID) = 0;
    virtual void pageWasNotifiedOfNotificationPermission() = 0;
    virtual void requestPermission(WebCore::SecurityOriginData&&, CompletionHandler<void(bool)>&&) = 0;
    virtual void setAppBadge(const WebCore::SecurityOriginData&, std::optional<uint64_t> badge) = 0;
    virtual void getPermissionState(WebCore::SecurityOriginData&&, CompletionHandler<void(WebCore::PushPermissionState)>&&) = 0;
    virtual void getPermissionStateSync(WebCore::SecurityOriginData&&, CompletionHandler<void(WebCore::PushPermissionState)>&&) = 0;
    virtual std::optional<SharedPreferencesForWebProcess> sharedPreferencesForWebProcess(const IPC::Connection&) const = 0;

    // IPC::MessageReceiver
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&);
    bool didReceiveSyncMessage(IPC::Connection&, IPC::Decoder&, UniqueRef<IPC::Encoder>&);
};

} // namespace WebKit
