/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 15, 2025.
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
#ifndef NotificationPermissionRequestManager_h
#define NotificationPermissionRequestManager_h

#include <WebCore/NotificationClient.h>
#include <WebCore/NotificationPermissionCallback.h>
#include <WebCore/SecurityOriginData.h>
#include <wtf/HashMap.h>
#include <wtf/RefCounted.h>
#include <wtf/RefPtr.h>
#include <wtf/text/WTFString.h>

namespace WebCore {
class Notification;
}

namespace WebKit {

class WebPage;

/// FIXME: Need to keep a queue of pending notifications which permission is still being requested.
class NotificationPermissionRequestManager : public RefCounted<NotificationPermissionRequestManager> {
public:
    static Ref<NotificationPermissionRequestManager> create(WebPage*);
    ~NotificationPermissionRequestManager();

    using Permission = WebCore::NotificationClient::Permission;
    using PermissionHandler = WebCore::NotificationClient::PermissionHandler;

#if ENABLE(NOTIFICATIONS)
    void startRequest(const WebCore::SecurityOriginData&, PermissionHandler&&);
#endif
    
    Permission permissionLevel(const WebCore::SecurityOriginData&);

    // For testing purposes only.
    void setPermissionLevelForTesting(const String& originString, bool allowed);
    void removeAllPermissionsForTesting();
    
private:
    NotificationPermissionRequestManager(WebPage*);

#if ENABLE(NOTIFICATIONS)
    using PermissionHandlers = Vector<PermissionHandler>;
    static void callPermissionHandlersWith(PermissionHandlers&, Permission);

    HashMap<WebCore::SecurityOriginData, PermissionHandlers> m_requestsPerOrigin;
    WeakPtr<WebPage> m_page;
#endif
};

inline bool isRequestIDValid(uint64_t id)
{
    // This check makes sure that the ID is not equal to values needed by
    // HashMap for bucketing.
    return id && id != static_cast<uint64_t>(-1);
}

} // namespace WebKit

#endif // NotificationPermissionRequestManager_h
