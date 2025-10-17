/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 16, 2023.
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
#include "config.h"
#include "MessageEvent.h"

#include "Blob.h"
#include "EventNames.h"
#include "JSMessageEvent.h"
#include <JavaScriptCore/JSCInlines.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

using namespace JSC;

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(MessageEvent);

MessageEvent::MessageEvent()
    : Event(EventInterfaceType::MessageEvent)
{
}

inline MessageEvent::MessageEvent(const AtomString& type, Init&& initializer, IsTrusted isTrusted)
    : Event(EventInterfaceType::MessageEvent, type, initializer, isTrusted)
    , m_data(JSValueTag { })
    , m_origin(initializer.origin)
    , m_lastEventId(initializer.lastEventId)
    , m_source(WTFMove(initializer.source))
    , m_ports(WTFMove(initializer.ports))
    , m_jsData(initializer.data)
{
}

inline MessageEvent::MessageEvent(const AtomString& type, DataType&& data, const String& origin, const String& lastEventId, std::optional<MessageEventSource>&& source, Vector<Ref<MessagePort>>&& ports)
    : Event(EventInterfaceType::MessageEvent, type, CanBubble::No, IsCancelable::No)
    , m_data(WTFMove(data))
    , m_origin(origin)
    , m_lastEventId(lastEventId)
    , m_source(WTFMove(source))
    , m_ports(WTFMove(ports))
{
}

auto MessageEvent::create(JSC::JSGlobalObject& globalObject, Ref<SerializedScriptValue>&& data, const String& origin, const String& lastEventId, std::optional<MessageEventSource>&& source, Vector<Ref<MessagePort>>&& ports) -> MessageEventWithStrongData
{
    auto& vm = globalObject.vm();
    Locker<JSC::JSLock> locker(vm.apiLock());
    auto catchScope = DECLARE_CATCH_SCOPE(vm);

    bool didFail = false;

    auto deserialized = data->deserialize(globalObject, &globalObject, ports, SerializationErrorMode::NonThrowing, &didFail);
    if (UNLIKELY(catchScope.exception()))
        deserialized = jsUndefined();
    JSC::Strong<JSC::Unknown> strongData(vm, deserialized);

    auto& eventType = didFail ? eventNames().messageerrorEvent : eventNames().messageEvent;
    Ref event = adoptRef(*new MessageEvent(eventType, MessageEvent::JSValueTag { }, origin, lastEventId, WTFMove(source), WTFMove(ports)));
    JSC::Strong<JSC::JSObject> strongWrapper(vm, JSC::jsCast<JSC::JSObject*>(toJS(&globalObject, JSC::jsCast<JSDOMGlobalObject*>(&globalObject), event.get())));
    event->jsData().set(vm, strongWrapper.get(), deserialized);

    return MessageEventWithStrongData { event, WTFMove(strongWrapper) };
}

Ref<MessageEvent> MessageEvent::create(const AtomString& type, DataType&& data, const String& origin, const String& lastEventId, std::optional<MessageEventSource>&& source, Vector<Ref<MessagePort>>&& ports)
{
    return adoptRef(*new MessageEvent(type, WTFMove(data), origin, lastEventId, WTFMove(source), WTFMove(ports)));
}

Ref<MessageEvent> MessageEvent::create(DataType&& data, const String& origin, const String& lastEventId, std::optional<MessageEventSource>&& source, Vector<Ref<MessagePort>>&& ports)
{
    return create(eventNames().messageEvent, WTFMove(data), origin, lastEventId, WTFMove(source), WTFMove(ports));
}

Ref<MessageEvent> MessageEvent::createForBindings()
{
    return adoptRef(*new MessageEvent);
}

Ref<MessageEvent> MessageEvent::create(const AtomString& type, Init&& initializer, IsTrusted isTrusted)
{
    return adoptRef(*new MessageEvent(type, WTFMove(initializer), isTrusted));
}

MessageEvent::~MessageEvent() = default;

void MessageEvent::initMessageEvent(const AtomString& type, bool canBubble, bool cancelable, JSValue data, const String& origin, const String& lastEventId, std::optional<MessageEventSource>&& source, Vector<Ref<MessagePort>>&& ports)
{
    if (isBeingDispatched())
        return;

    initEvent(type, canBubble, cancelable);

    {
        Locker locker { m_concurrentDataAccessLock };
        m_data = JSValueTag { };
    }
    // FIXME: This code is wrong: we should emit a write-barrier. Otherwise, GC can collect it.
    // https://bugs.webkit.org/show_bug.cgi?id=236353
    m_jsData.setWeakly(data);
    m_cachedData.clear();
    m_origin = origin;
    m_lastEventId = lastEventId;
    m_source = WTFMove(source);
    m_ports = WTFMove(ports);
    m_cachedPorts.clear();
}

size_t MessageEvent::memoryCost() const
{
    Locker locker { m_concurrentDataAccessLock };
    return WTF::switchOn(m_data, [] (JSValueTag) -> size_t {
        return 0;
    }, [] (const Ref<SerializedScriptValue>& data) -> size_t {
        return data->memoryCost();
    }, [] (const String& string) -> size_t {
        return string.sizeInBytes();
    }, [] (const Ref<Blob>& blob) -> size_t {
        return blob->memoryCost();
    }, [] (const Ref<ArrayBuffer>& buffer) -> size_t {
        return buffer->byteLength();
    });
}

} // namespace WebCore
