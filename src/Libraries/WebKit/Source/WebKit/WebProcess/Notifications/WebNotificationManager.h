/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 15, 2023.
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
#include "WebProcessSupplement.h"
#include <WebCore/NotificationClient.h>
#include <WebCore/ScriptExecutionContextIdentifier.h>
#include <optional>
#include <wtf/CompletionHandler.h>
#include <wtf/HashMap.h>
#include <wtf/Noncopyable.h>
#include <wtf/RefPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/UUID.h>
#include <wtf/Vector.h>
#include <wtf/WeakRef.h>
#include <wtf/text/StringHash.h>

namespace WebCore {
class SecurityOrigin;
class SecurityOriginData;

struct NotificationData;
}

namespace WebKit {

class WebPage;
class WebProcess;

class WebNotificationManager : public WebProcessSupplement, public IPC::MessageReceiver {
    WTF_MAKE_TZONE_ALLOCATED(WebNotificationManager);
    WTF_MAKE_NONCOPYABLE(WebNotificationManager);
public:
    explicit WebNotificationManager(WebProcess&);
    ~WebNotificationManager();

    void ref() const final;
    void deref() const final;

    static ASCIILiteral supplementName();
    
    bool show(WebCore::NotificationData&&, RefPtr<WebCore::NotificationResources>&&, WebPage*, CompletionHandler<void()>&&);
    void cancel(WebCore::NotificationData&&, WebPage*);

    void requestPermission(WebCore::SecurityOriginData&&, RefPtr<WebPage>, CompletionHandler<void(bool)>&&);

    // This callback comes from WebCore, not messaged from the UI process.
    void didDestroyNotification(WebCore::NotificationData&&, WebPage*);

    void didUpdateNotificationDecision(const String& originString, bool allowed);

    // Looks in local cache for permission. If not found, returns DefaultDenied.
    WebCore::NotificationClient::Permission policyForOrigin(const String& originString, WebPage* = nullptr) const;

    void removeAllPermissionsForTesting();

private:
    // WebProcessSupplement
    void initialize(const WebProcessCreationParameters&) override;

    // IPC::MessageReceiver
    // Implemented in generated WebNotificationManagerMessageReceiver.cpp
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) override;
    
    void didShowNotification(const WTF::UUID& notificationID);
    void didClickNotification(const WTF::UUID& notificationID);
    void didCloseNotifications(const Vector<WTF::UUID>& notificationIDs);
    void didRemoveNotificationDecisions(const Vector<String>& originStrings);

    WeakRef<WebProcess> m_process;
#if ENABLE(NOTIFICATIONS)
    HashMap<WTF::UUID, WebCore::ScriptExecutionContextIdentifier> m_nonPersistentNotificationsContexts;
    HashMap<String, bool> m_permissionsMap;
#endif
};

} // namespace WebKit
