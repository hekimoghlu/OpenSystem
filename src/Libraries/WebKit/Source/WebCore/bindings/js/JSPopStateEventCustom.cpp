/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 21, 2025.
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
#include "JSPopStateEvent.h"

#include "DOMWrapperWorld.h"
#include "JSHistory.h"
#include <JavaScriptCore/HeapInlines.h>
#include <JavaScriptCore/JSCJSValueInlines.h>

namespace WebCore {
using namespace JSC;

JSValue JSPopStateEvent::state(JSGlobalObject& lexicalGlobalObject) const
{
    if (m_state) {
        // We cannot use a cached object if we are in a different world than the one it was created in.
        if (isWorldCompatible(lexicalGlobalObject, m_state.get()))
            return m_state.get();
        ASSERT_NOT_REACHED();
    }

    // Save the lexicalGlobalObject value to the m_state member of a JSPopStateEvent, and return it, for convenience.
    auto cacheState = [&lexicalGlobalObject, this] (JSC::JSValue eventState) {
        m_state.set(lexicalGlobalObject.vm(), this, eventState);
        return eventState;
    };

    PopStateEvent& event = wrapped();

    if (event.state()) {
        JSC::JSValue eventState = event.state().getValue();
        // We need to make sure a PopStateEvent does not leak objects in its lexicalGlobalObject property across isolated DOM worlds.
        // Ideally, we would check that the worlds have different privileges but that's not possible yet.
        if (!isWorldCompatible(lexicalGlobalObject, eventState)) {
            if (auto serializedValue = event.trySerializeState(lexicalGlobalObject))
                eventState = serializedValue->deserialize(lexicalGlobalObject, globalObject());
            else
                eventState = jsNull();
        }
        return cacheState(eventState);
    }
    
    History* history = event.history();
    if (!history || !event.serializedState())
        return cacheState(jsNull());

    // There's no cached value from a previous invocation, nor a lexicalGlobalObject value was provided by the
    // event, but there is a history object, so first we need to see if the lexicalGlobalObject object has been
    // deserialized through the history object already.
    // The current history lexicalGlobalObject object might've changed in the meantime, so we need to take care
    // of using the correct one, and always share the same deserialization with history.lexicalGlobalObject.

    bool isSameState = history->isSameAsCurrentState(event.serializedState());
    JSValue result;

    if (isSameState) {
        JSHistory* jsHistory = jsCast<JSHistory*>(toJS(&lexicalGlobalObject, globalObject(), *history).asCell());
        result = jsHistory->state(lexicalGlobalObject);
    } else
        result = event.serializedState()->deserialize(lexicalGlobalObject, globalObject());

    return cacheState(result);
}

template<typename Visitor>
void JSPopStateEvent::visitAdditionalChildren(Visitor& visitor)
{
    wrapped().state().visit(visitor);
}

DEFINE_VISIT_ADDITIONAL_CHILDREN(JSPopStateEvent);

} // namespace WebCore
