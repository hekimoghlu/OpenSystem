/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 13, 2024.
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

#include "ExtendableEvent.h"
#include "ExtendableEventInit.h"
#include "JSValueInWrappedObject.h"
#include "MessagePort.h"
#include "ServiceWorker.h"
#include "ServiceWorkerClient.h"
#include <variant>

namespace JSC {
class CallFrame;
class JSValue;
}

namespace WebCore {

class MessagePort;
class ServiceWorker;
class ServiceWorkerClient;

using ExtendableMessageEventSource = std::variant<RefPtr<ServiceWorkerClient>, RefPtr<ServiceWorker>, RefPtr<MessagePort>>;

class ExtendableMessageEvent final : public ExtendableEvent {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(ExtendableMessageEvent);
public:
    struct Init : ExtendableEventInit {
        JSC::JSValue data;
        String origin;
        String lastEventId;
        std::optional<ExtendableMessageEventSource> source;
        Vector<Ref<MessagePort>> ports;
    };

    struct ExtendableMessageEventWithStrongData {
        Ref<ExtendableMessageEvent> event;
        JSC::Strong<JSC::JSObject> strongWrapper; // Keep the wrapper alive until the event is fired, since it is what keeps `data` alive.
    };

    static ExtendableMessageEventWithStrongData create(JSC::JSGlobalObject&, const AtomString& type, const Init&, IsTrusted = IsTrusted::No);
    static ExtendableMessageEventWithStrongData create(JSC::JSGlobalObject&, Vector<Ref<MessagePort>>&&, Ref<SerializedScriptValue>&&, const String& origin, const String& lastEventId, std::optional<ExtendableMessageEventSource>&&);

    ~ExtendableMessageEvent();

    JSValueInWrappedObject& data() { return m_data; }
    JSValueInWrappedObject& cachedPorts() { return m_cachedPorts; }

    const String& origin() const { return m_origin; }
    const String& lastEventId() const { return m_lastEventId; }
    const std::optional<ExtendableMessageEventSource>& source() const { return m_source; }
    const Vector<Ref<MessagePort>>& ports() const { return m_ports; }

private:
    ExtendableMessageEvent(const AtomString&, const Init&, IsTrusted);
    ExtendableMessageEvent(const AtomString&, const String& origin, const String& lastEventId, std::optional<ExtendableMessageEventSource>&&, Vector<Ref<MessagePort>>&&);

    JSValueInWrappedObject m_data;
    String m_origin;
    String m_lastEventId;
    std::optional<ExtendableMessageEventSource> m_source;
    Vector<Ref<MessagePort>> m_ports;
    JSValueInWrappedObject m_cachedPorts;
};

} // namespace WebCore
