/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 30, 2023.
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

#if ENABLE(NOTIFICATIONS)

#include <WebCore/NotificationClient.h>
#include <WebCore/SecurityOriginData.h>
#include <wtf/HashSet.h>
#include <wtf/TZoneMalloc.h>

namespace WebKit {

class WebPage;

class WebNotificationClient final : public WebCore::NotificationClient {
    WTF_MAKE_TZONE_ALLOCATED(WebNotificationClient);
public:
    WebNotificationClient(WebPage*);
    virtual ~WebNotificationClient();
    void clearNotificationPermissionState();

private:
    bool show(WebCore::ScriptExecutionContext&, WebCore::NotificationData&&, RefPtr<WebCore::NotificationResources>&&, CompletionHandler<void()>&&) final;
    void cancel(WebCore::NotificationData&&) final;
    void notificationObjectDestroyed(WebCore::NotificationData&&) final;
    void notificationControllerDestroyed() final;
    void requestPermission(WebCore::ScriptExecutionContext&, PermissionHandler&&) final;
    WebCore::NotificationClient::Permission checkPermission(WebCore::ScriptExecutionContext*) final;

    HashSet<WebCore::SecurityOriginData> m_notificationPermissionRequesters;
    WeakPtr<WebPage> m_page;
};

} // namespace WebKit

#endif // ENABLE(NOTIFICATIONS)
