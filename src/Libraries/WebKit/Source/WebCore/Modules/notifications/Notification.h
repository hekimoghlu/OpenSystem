/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 17, 2024.
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

#include "ActiveDOMObject.h"
#include "ContextDestructionObserverInlines.h"
#include "EventTarget.h"
#include "NotificationDirection.h"
#include "NotificationPayload.h"
#include "NotificationPermission.h"
#include "NotificationResources.h"
#include "ScriptExecutionContextIdentifier.h"
#include "SerializedScriptValue.h"
#include <wtf/CompletionHandler.h>
#include <wtf/URL.h>
#include <wtf/UUID.h>
#include "WritingMode.h"

namespace WebCore {

class DeferredPromise;
class Document;
class NotificationClient;
class NotificationPermissionCallback;
class NotificationResourcesLoader;

struct NotificationData;

class Notification final : public RefCounted<Notification>, public ActiveDOMObject, public EventTarget {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED_EXPORT(Notification, WEBCORE_EXPORT);
public:
    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    using Permission = NotificationPermission;
    using Direction = NotificationDirection;

    struct Options {
        Direction dir;
        String lang;
        String body;
        String tag;
        String icon;
        JSC::JSValue data;
        RefPtr<SerializedScriptValue> serializedData;
        RefPtr<JSON::Value> jsonData;
        std::optional<bool> silent;
#if ENABLE(DECLARATIVE_WEB_PUSH)
        String navigate;
#endif
    };
    // For JS constructor only.
    static ExceptionOr<Ref<Notification>> create(ScriptExecutionContext&, String&& title, Options&&);

    static ExceptionOr<Ref<Notification>> createForServiceWorker(ScriptExecutionContext&, String&& title, Options&&, const URL&);
    static Ref<Notification> create(ScriptExecutionContext&, NotificationData&&);
    static Ref<Notification> create(ScriptExecutionContext&, const URL& registrationURL, const NotificationPayload&);

    WEBCORE_EXPORT virtual ~Notification();

    void show(CompletionHandler<void()>&& = [] { });
    void close();

#if ENABLE(DECLARATIVE_WEB_PUSH)
    const URL& navigate() const { return m_navigate; }
#endif
    const String& title() const { return m_title; }
    Direction dir() const { return m_direction; }
    const String& body() const { return m_body; }
    const String& lang() const { return m_lang; }
    const String& tag() const { return m_tag; }
    const URL& icon() const { return m_icon; }
    JSC::JSValue dataForBindings(JSC::JSGlobalObject&);
    std::optional<bool> silent() const { return m_silent; }

    TextDirection direction() const { return m_direction == Direction::Rtl ? TextDirection::RTL : TextDirection::LTR; }

    WEBCORE_EXPORT void dispatchClickEvent();
    WEBCORE_EXPORT void dispatchCloseEvent();
    WEBCORE_EXPORT void dispatchErrorEvent();
    WEBCORE_EXPORT void dispatchShowEvent();

    WEBCORE_EXPORT void finalize();

    static Permission permission(ScriptExecutionContext&);
    static void requestPermission(Document&, RefPtr<NotificationPermissionCallback>&&, Ref<DeferredPromise>&&);

    ScriptExecutionContext* scriptExecutionContext() const final { return ActiveDOMObject::scriptExecutionContext(); }

    WEBCORE_EXPORT NotificationData data() const;
    RefPtr<NotificationResources> resources() const { return m_resources; }

    void markAsShown();
    void showSoon();

    WTF::UUID identifier() const { return m_identifier; }

    bool isPersistent() const { return !m_serviceWorkerRegistrationURL.isNull(); }

    WEBCORE_EXPORT static void ensureOnNotificationThread(ScriptExecutionContextIdentifier, WTF::UUID notificationIdentifier, Function<void(Notification*)>&&);
    WEBCORE_EXPORT static void ensureOnNotificationThread(const NotificationData&, Function<void(Notification*)>&&);

private:
    Notification(ScriptExecutionContext&, WTF::UUID, const String& title, Options&&, Ref<SerializedScriptValue>&&);

    NotificationClient* clientFromContext();
    enum EventTargetInterfaceType eventTargetInterface() const final { return EventTargetInterfaceType::Notification; }

    void stopResourcesLoader();

    // ActiveDOMObject
    void suspend(ReasonForSuspension);
    void stop() final;
    bool virtualHasPendingActivity() const final;

    // EventTarget
    void refEventTarget() final { ref(); }
    void derefEventTarget() final { deref(); }
    void eventListenersDidChange() final;

    WTF::UUID m_identifier;

#if ENABLE(DECLARATIVE_WEB_PUSH)
    URL m_navigate;
#endif
    String m_title;
    Direction m_direction;
    String m_lang;
    String m_body;
    String m_tag;
    URL m_icon;
    Ref<SerializedScriptValue> m_dataForBindings;
    std::optional<bool> m_silent;

    enum State { Idle, Showing, Closed };
    State m_state { Idle };
    bool m_hasRelevantEventListener { false };

    enum class NotificationSource : uint8_t {
        DedicatedWorker,
        Document,
        ServiceWorker,
    };
    NotificationSource m_notificationSource;
    URL m_serviceWorkerRegistrationURL;
    std::unique_ptr<NotificationResourcesLoader> m_resourcesLoader;
    RefPtr<NotificationResources> m_resources;
};

} // namespace WebCore

#endif // ENABLE(NOTIFICATIONS)
