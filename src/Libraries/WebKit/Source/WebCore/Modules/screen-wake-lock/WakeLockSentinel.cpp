/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 26, 2024.
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
#include "WakeLockSentinel.h"

#include "Document.h"
#include "EventNames.h"
#include "Exception.h"
#include "JSDOMPromiseDeferred.h"
#include "WakeLockManager.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(WakeLockSentinel);

WakeLockSentinel::WakeLockSentinel(Document& document, WakeLockType type)
    : ActiveDOMObject(&document)
    , m_type(type)
{
}

void WakeLockSentinel::release(Ref<DeferredPromise>&& promise)
{
    if (!m_wasReleased) {
        if (RefPtr document = downcast<Document>(scriptExecutionContext())) {
            Ref wakeLockManagerRef { document->wakeLockManager() };
            Ref { *this }->release(wakeLockManagerRef.get());
        }
    }
    promise->resolve();
}

// https://www.w3.org/TR/screen-wake-lock/#dfn-release-a-wake-lock
void WakeLockSentinel::release(WakeLockManager& manager)
{
    manager.removeWakeLock(*this);

    m_wasReleased = true;

    if (scriptExecutionContext() && !scriptExecutionContext()->activeDOMObjectsAreStopped())
        dispatchEvent(Event::create(eventNames().releaseEvent, Event::CanBubble::No, Event::IsCancelable::No));
}

// https://www.w3.org/TR/screen-wake-lock/#garbage-collection
bool WakeLockSentinel::virtualHasPendingActivity() const
{
    return m_hasReleaseEventListener && !m_wasReleased;
}

void WakeLockSentinel::eventListenersDidChange()
{
    m_hasReleaseEventListener = hasEventListeners(eventNames().releaseEvent);
}

} // namespace WebCore
