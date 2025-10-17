/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 21, 2021.
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

namespace WebCore {

class History;
class SerializedScriptValue;

class PopStateEvent final : public Event {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(PopStateEvent);
public:
    virtual ~PopStateEvent();
    static Ref<PopStateEvent> create(RefPtr<SerializedScriptValue>&&, History*);

    struct Init : EventInit {
        JSC::JSValue state;
        bool hasUAVisualTransition { false };
    };

    static Ref<PopStateEvent> create(const AtomString&, const Init&, IsTrusted = IsTrusted::No);
    static Ref<PopStateEvent> createForBindings();

    const JSValueInWrappedObject& state() const { return m_state; }
    SerializedScriptValue* serializedState() const { return m_serializedState.get(); }

    RefPtr<SerializedScriptValue> trySerializeState(JSC::JSGlobalObject&);
    
    History* history() const { return m_history.get(); }

    bool hasUAVisualTransition() const { return m_hasUAVisualTransition; }
    void setHasUAVisualTransition(bool hasUAVisualTransition) { m_hasUAVisualTransition = hasUAVisualTransition; }

private:
    PopStateEvent();
    PopStateEvent(const AtomString&, const Init&, IsTrusted);
    PopStateEvent(RefPtr<SerializedScriptValue>&&, History*);

    JSValueInWrappedObject m_state;
    RefPtr<SerializedScriptValue> m_serializedState;
    bool m_triedToSerialize { false };
    bool m_hasUAVisualTransition { false };
    RefPtr<History> m_history;
};

} // namespace WebCore
