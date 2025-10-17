/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 2, 2025.
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
#include "ExtendableEvent.h"

#include "JSDOMGlobalObject.h"
#include "JSDOMPromise.h"
#include "ScriptExecutionContext.h"
#include <JavaScriptCore/Microtask.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(ExtendableEvent);

ExtendableEvent::ExtendableEvent(enum EventInterfaceType eventInterface, const AtomString& type, const ExtendableEventInit& initializer, IsTrusted isTrusted)
    : Event(eventInterface, type, initializer, isTrusted)
{
}

ExtendableEvent::ExtendableEvent(enum EventInterfaceType eventInterface, const AtomString& type, CanBubble canBubble, IsCancelable cancelable)
    : Event(eventInterface, type, canBubble, cancelable)
{
}

ExtendableEvent::~ExtendableEvent()
{
}

// https://w3c.github.io/ServiceWorker/#dom-extendableevent-waituntil
ExceptionOr<void> ExtendableEvent::waitUntil(Ref<DOMPromise>&& promise)
{
    if (!isTrusted())
        return Exception { ExceptionCode::InvalidStateError, "Event is not trusted"_s };

    // If the pending promises count is zero and the dispatch flag is unset, throw an "InvalidStateError" DOMException.
    if (!m_pendingPromiseCount && !isBeingDispatched())
        return Exception { ExceptionCode::InvalidStateError, "Event is no longer being dispatched and has no pending promises"_s };

    addExtendLifetimePromise(WTFMove(promise));
    return { };
}

void ExtendableEvent::addExtendLifetimePromise(Ref<DOMPromise>&& promise)
{
    promise->whenSettled([this, protectedThis = Ref { *this }, settledPromise = promise.ptr()] () mutable {
        auto& globalObject = *settledPromise->globalObject();
        auto* context = globalObject.scriptExecutionContext();
        if (!context)
            return;
        context->eventLoop().queueMicrotask([this, protectedThis = WTFMove(protectedThis), settledPromise = WTFMove(settledPromise)]() mutable {
            --m_pendingPromiseCount;

            // FIXME: Let registration be the context object's relevant global object's associated service worker's containing service worker registration.
            // FIXME: If registration's uninstalling flag is set, invoke Try Clear Registration with registration.
            // FIXME: If registration is not null, invoke Try Activate with registration.

            auto* context = settledPromise->globalObject()->scriptExecutionContext();
            if (!context)
                return;
            context->postTask([this, protectedThis = WTFMove(protectedThis)] (ScriptExecutionContext&) mutable {
                if (m_pendingPromiseCount)
                    return;

                auto settledPromises = WTFMove(m_extendLifetimePromises);
                if (auto handler = WTFMove(m_whenAllExtendLifetimePromisesAreSettledHandler))
                    handler(WTFMove(settledPromises));
            });
        });
    });

    m_extendLifetimePromises.add(WTFMove(promise));
    ++m_pendingPromiseCount;
}

void ExtendableEvent::whenAllExtendLifetimePromisesAreSettled(Function<void(HashSet<Ref<DOMPromise>>&&)>&& handler)
{
    ASSERT_WITH_MESSAGE(target(), "Event has not been dispatched yet");
    ASSERT(!m_whenAllExtendLifetimePromisesAreSettledHandler);

    if (!m_pendingPromiseCount) {
        handler(WTFMove(m_extendLifetimePromises));
        return;
    }

    m_whenAllExtendLifetimePromisesAreSettledHandler = WTFMove(handler);
}

} // namespace WebCore
