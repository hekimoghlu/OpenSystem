/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 27, 2021.
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
#ifndef WebNotificationProvider_h
#define WebNotificationProvider_h

#include "APIClient.h"
#include "APINotificationProvider.h"
#include "WKNotificationProvider.h"
#include <wtf/Forward.h>
#include <wtf/TZoneMalloc.h>

namespace API {
template<> struct ClientTraits<WKNotificationProviderBase> {
    typedef std::tuple<WKNotificationProviderV0> Versions;
};
}

namespace WebKit {

class WebNotification;
class WebNotificationManagerProxy;
class WebPageProxy;

class WebNotificationProvider : public API::NotificationProvider, public API::Client<WKNotificationProviderBase> {
    WTF_MAKE_TZONE_ALLOCATED(WebNotificationProvider);
public:
    explicit WebNotificationProvider(const WKNotificationProviderBase*);

    bool show(WebPageProxy*, WebNotification&, RefPtr<WebCore::NotificationResources>&&) override;
    void cancel(WebNotification&) override;
    void didDestroyNotification(WebNotification&) override;
    void clearNotifications(const Vector<WebNotificationIdentifier>&) override;

    void addNotificationManager(WebNotificationManagerProxy&) override;
    void removeNotificationManager(WebNotificationManagerProxy&) override;

    HashMap<WTF::String, bool> notificationPermissions() override;
};

} // namespace WebKit

#endif // WebNotificationProvider_h
