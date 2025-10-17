/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 24, 2023.
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

#include <wtf/Forward.h>
#include <wtf/HashMap.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/StringHash.h>

namespace WebKit {
class WebNotification;
class WebNotificationManagerProxy;
class WebPageProxy;

enum class WebNotificationIdentifierType;
using WebNotificationIdentifier = ObjectIdentifier<WebNotificationIdentifierType>;
}

namespace WebCore {
class NotificationResources;
}

namespace API {

class NotificationProvider {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(NotificationProvider);
public:
    virtual ~NotificationProvider() = default;

    virtual bool show(WebKit::WebPageProxy*, WebKit::WebNotification&, RefPtr<WebCore::NotificationResources>&&) { return false; }
    virtual void cancel(WebKit::WebNotification&) { }
    virtual void didDestroyNotification(WebKit::WebNotification&) { }
    virtual void clearNotifications(const Vector<WebKit::WebNotificationIdentifier>&) { }

    virtual void addNotificationManager(WebKit::WebNotificationManagerProxy&) { }
    virtual void removeNotificationManager(WebKit::WebNotificationManagerProxy&) { }

    virtual HashMap<WTF::String, bool> notificationPermissions() { return HashMap<WTF::String, bool>(); };
};

} // namespace API
