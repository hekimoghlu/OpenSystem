/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 26, 2025.
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
#include "ExtendableEventInit.h"
#include <wtf/WeakPtr.h>

namespace WebCore {

class DOMPromise;

class ExtendableEvent : public Event {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(ExtendableEvent);
public:
    static Ref<ExtendableEvent> create(const AtomString& type, const ExtendableEventInit& initializer, IsTrusted isTrusted = IsTrusted::No)
    {
        return adoptRef(*new ExtendableEvent(EventInterfaceType::ExtendableEvent, type, initializer, isTrusted));
    }

    ~ExtendableEvent();

    ExceptionOr<void> waitUntil(Ref<DOMPromise>&&);
    unsigned pendingPromiseCount() const { return m_pendingPromiseCount; }

    WEBCORE_EXPORT void whenAllExtendLifetimePromisesAreSettled(Function<void(HashSet<Ref<DOMPromise>>&&)>&&);

protected:
    WEBCORE_EXPORT ExtendableEvent(enum EventInterfaceType, const AtomString&, const ExtendableEventInit&, IsTrusted);
    ExtendableEvent(enum EventInterfaceType, const AtomString&, CanBubble, IsCancelable);

    void addExtendLifetimePromise(Ref<DOMPromise>&&);

private:
    unsigned m_pendingPromiseCount { 0 };
    HashSet<Ref<DOMPromise>> m_extendLifetimePromises;
    Function<void(HashSet<Ref<DOMPromise>>&&)> m_whenAllExtendLifetimePromisesAreSettledHandler;
};

} // namespace WebCore
