/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 4, 2021.
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
#include "PopStateEvent.h"

#include "EventNames.h"
#include "History.h"
#include <JavaScriptCore/JSCInlines.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(PopStateEvent);

PopStateEvent::PopStateEvent()
    : Event(EventInterfaceType::PopStateEvent)
{
}

PopStateEvent::PopStateEvent(const AtomString& type, const Init& initializer, IsTrusted isTrusted)
    : Event(EventInterfaceType::PopStateEvent, type, initializer, isTrusted)
    , m_state(initializer.state)
    , m_hasUAVisualTransition(initializer.hasUAVisualTransition)
{
}

PopStateEvent::PopStateEvent(RefPtr<SerializedScriptValue>&& serializedState, History* history)
    : Event(EventInterfaceType::PopStateEvent, eventNames().popstateEvent, CanBubble::No, IsCancelable::No)
    , m_serializedState(WTFMove(serializedState))
    , m_history(history)
{
}

PopStateEvent::~PopStateEvent() = default;

Ref<PopStateEvent> PopStateEvent::create(RefPtr<SerializedScriptValue>&& serializedState, History* history)
{
    return adoptRef(*new PopStateEvent(WTFMove(serializedState), history));
}

Ref<PopStateEvent> PopStateEvent::create(const AtomString& type, const Init& initializer, IsTrusted isTrusted)
{
    return adoptRef(*new PopStateEvent(type, initializer, isTrusted));
}

Ref<PopStateEvent> PopStateEvent::createForBindings()
{
    return adoptRef(*new PopStateEvent);
}

RefPtr<SerializedScriptValue> PopStateEvent::trySerializeState(JSC::JSGlobalObject& executionState)
{
    ASSERT(m_state);
    
    if (!m_serializedState && !m_triedToSerialize) {
        m_serializedState = SerializedScriptValue::create(executionState, m_state.getValue(), SerializationForStorage::No, SerializationErrorMode::NonThrowing);
        m_triedToSerialize = true;
    }
    
    return m_serializedState;
}

} // namespace WebCore
