/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 3, 2025.
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

#include "JSDOMPromise.h"
#include "NavigationHistoryEntry.h"
#include "NavigationNavigationType.h"

namespace WebCore {

class DOMPromise;

class NavigationTransition final : public RefCounted<NavigationTransition> {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(NavigationTransition);
public:
    static Ref<NavigationTransition> create(NavigationNavigationType type, Ref<NavigationHistoryEntry>&& fromEntry, Ref<DeferredPromise>&& finished) { return adoptRef(*new NavigationTransition(type, WTFMove(fromEntry), WTFMove(finished))); };

    NavigationNavigationType navigationType() { return m_navigationType; };
    NavigationHistoryEntry& from() { return m_from; };
    DOMPromise* finished();

    void resolvePromise();
    void rejectPromise(Exception&, JSC::JSValue exceptionObject);

private:
    explicit NavigationTransition(NavigationNavigationType, Ref<NavigationHistoryEntry>&& fromEntry, Ref<DeferredPromise>&& finished);

    NavigationNavigationType m_navigationType;
    Ref<NavigationHistoryEntry> m_from;
    Ref<DeferredPromise> m_finished;
    RefPtr<DOMPromise> m_finishedDOMPromise;
};

} // namespace WebCore
