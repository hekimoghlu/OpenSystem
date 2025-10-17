/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 19, 2025.
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

#include "WebNotificationIdentifier.h"
#include <wtf/HashMap.h>
#include <wtf/HashSet.h>
#include <wtf/OptionSet.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/glib/GRefPtr.h>
#include <wtf/text/WTFString.h>

typedef struct _GDBusProxy GDBusProxy;
typedef struct _GVariant GVariant;

namespace WebCore {
class NotificationResources;
}

namespace WebKit {

class WebNotification;

class NotificationService {
    WTF_MAKE_NONCOPYABLE(NotificationService);
    WTF_MAKE_TZONE_ALLOCATED(NotificationService);
    friend LazyNeverDestroyed<NotificationService>;
public:
    static NotificationService& singleton();

    bool showNotification(const WebNotification&, const RefPtr<WebCore::NotificationResources>&);
    void cancelNotification(WebNotificationIdentifier);

    class Observer {
    public:
        virtual void didClickNotification(WebNotificationIdentifier) = 0;
        virtual void didCloseNotification(WebNotificationIdentifier) = 0;
    };
    void addObserver(Observer&);
    void removeObserver(Observer&);

private:
    NotificationService();

    struct Notification {
        uint32_t id { 0 };
        String portalID;
        String tag;
        String iconURL;
    };

    enum class Capabilities : uint16_t {
        ActionIcons = 1 << 0,
        Actions = 1 << 1,
        Body = 1 << 2,
        BodyHyperlinks = 1 << 3,
        BodyImages = 1 << 4,
        BodyMarkup = 1 << 5,
        IconMulti = 1 << 6,
        IconStatic = 1 << 7,
        Persistence = 1 << 8,
        Sound =  1 << 9
    };
    void processCapabilities(GVariant*);

    void setNotificationID(WebNotificationIdentifier, uint32_t);
    std::optional<WebNotificationIdentifier> findNotification(uint32_t);
    std::optional<WebNotificationIdentifier> findNotification(const String&);

    static void handleSignal(GDBusProxy*, char*, char*, GVariant*, NotificationService*);
    void didClickNotification(std::optional<WebNotificationIdentifier>);
    void didCloseNotification(std::optional<WebNotificationIdentifier>);

    GRefPtr<GDBusProxy> m_proxy;
    OptionSet<Capabilities> m_capabilities;
    HashMap<WebNotificationIdentifier, Notification> m_notifications;
    HashSet<Observer*> m_observers;
};

} // namespace WebKit
