/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 10, 2022.
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

#include "Event.h"
#include "JSValueInWrappedObject.h"
#include "MessagePort.h"
#include "SerializedScriptValue.h"
#include "ServiceWorker.h"
#include "WindowProxy.h"
#include <variant>

namespace WebCore {

class Blob;

using MessageEventSource = std::variant<RefPtr<WindowProxy>, RefPtr<MessagePort>, RefPtr<ServiceWorker>>;

class MessageEvent final : public Event {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(MessageEvent);
public:
    struct JSValueTag { };
    using DataType = std::variant<JSValueTag, Ref<SerializedScriptValue>, String, Ref<Blob>, Ref<ArrayBuffer>>;
    static Ref<MessageEvent> create(const AtomString& type, DataType&&, const String& origin = { }, const String& lastEventId = { }, std::optional<MessageEventSource>&& = std::nullopt, Vector<Ref<MessagePort>>&& = { });
    static Ref<MessageEvent> create(DataType&&, const String& origin = { }, const String& lastEventId = { }, std::optional<MessageEventSource>&& = std::nullopt, Vector<Ref<MessagePort>>&& = { });
    static Ref<MessageEvent> createForBindings();

    struct MessageEventWithStrongData {
        Ref<MessageEvent> event;
        JSC::Strong<JSC::JSObject> strongWrapper; // Keep the wrapper alive until the event is fired, since it is what keeps `data` alive.
    };

    static MessageEventWithStrongData create(JSC::JSGlobalObject&, Ref<SerializedScriptValue>&&, const String& origin = { }, const String& lastEventId = { }, std::optional<MessageEventSource>&& = std::nullopt, Vector<Ref<MessagePort>>&& = { });

    struct Init : EventInit {
        JSC::JSValue data;
        String origin;
        String lastEventId;
        std::optional<MessageEventSource> source;
        Vector<Ref<MessagePort>> ports;
    };
    static Ref<MessageEvent> create(const AtomString& type, Init&&, IsTrusted = IsTrusted::No);

    virtual ~MessageEvent();

    void initMessageEvent(const AtomString& type, bool canBubble, bool cancelable, JSC::JSValue data, const String& origin, const String& lastEventId, std::optional<MessageEventSource>&&, Vector<Ref<MessagePort>>&&);

    const String& origin() const { return m_origin; }
    const String& lastEventId() const { return m_lastEventId; }
    const std::optional<MessageEventSource>& source() const { return m_source; }
    const Vector<Ref<MessagePort>>& ports() const { return m_ports; }

    const DataType& data() const
    {
        IGNORE_CLANG_WARNINGS_BEGIN("thread-safety-reference-return")
        return m_data;
        IGNORE_CLANG_WARNINGS_END
    }

    JSValueInWrappedObject& jsData() { return m_jsData; }
    JSValueInWrappedObject& cachedData() { return m_cachedData; }
    JSValueInWrappedObject& cachedPorts() { return m_cachedPorts; }

    size_t memoryCost() const;

private:
    MessageEvent();
    MessageEvent(const AtomString& type, Init&&, IsTrusted);
    MessageEvent(const AtomString& type, DataType&&, const String& origin, const String& lastEventId = { }, std::optional<MessageEventSource>&& = std::nullopt, Vector<Ref<MessagePort>>&& = { });

    DataType m_data WTF_GUARDED_BY_LOCK(m_concurrentDataAccessLock);
    String m_origin;
    String m_lastEventId;
    std::optional<MessageEventSource> m_source;
    Vector<Ref<MessagePort>> m_ports;

    JSValueInWrappedObject m_jsData;
    JSValueInWrappedObject m_cachedData;
    JSValueInWrappedObject m_cachedPorts;

    mutable Lock m_concurrentDataAccessLock;
};

} // namespace WebCore
