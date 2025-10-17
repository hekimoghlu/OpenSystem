/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 18, 2022.
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
#include "NavigationTransition.h"

#include "JSDOMPromiseDeferred.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(NavigationTransition);

NavigationTransition::NavigationTransition(NavigationNavigationType type, Ref<NavigationHistoryEntry>&& fromEntry, Ref<DeferredPromise>&& finished)
    : m_navigationType(type)
    , m_from(WTFMove(fromEntry))
    , m_finished(WTFMove(finished))
{
}

void NavigationTransition::resolvePromise()
{
    m_finished->resolve();
}

void NavigationTransition::rejectPromise(Exception& exception, JSC::JSValue exceptionObject)
{
    m_finished->reject(exception, RejectAsHandled::Yes, exceptionObject);
}

DOMPromise* NavigationTransition::finished()
{
    if (!m_finishedDOMPromise) {
        auto& promise = *jsCast<JSC::JSPromise*>(m_finished->promise());
        m_finishedDOMPromise = DOMPromise::create(*m_finished->globalObject(), promise);
    }

    return m_finishedDOMPromise.get();
}

} // namespace WebCore
