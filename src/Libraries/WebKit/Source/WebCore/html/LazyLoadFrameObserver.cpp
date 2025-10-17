/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 20, 2022.
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
#include "LazyLoadFrameObserver.h"

#include "DocumentInlines.h"
#include "HTMLIFrameElement.h"
#include "IntersectionObserverCallback.h"
#include "IntersectionObserverEntry.h"
#include "LocalFrame.h"
#include "RenderStyle.h"

#include <limits>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(LazyLoadFrameObserver);

class LazyFrameLoadIntersectionObserverCallback final : public IntersectionObserverCallback {
public:
    static Ref<LazyFrameLoadIntersectionObserverCallback> create(Document& document)
    {
        return adoptRef(*new LazyFrameLoadIntersectionObserverCallback(document));
    }

private:
    LazyFrameLoadIntersectionObserverCallback(Document& document)
        : IntersectionObserverCallback(&document)
    {
    }

    bool hasCallback() const final { return true; }

    CallbackResult<void> handleEvent(IntersectionObserver&, const Vector<Ref<IntersectionObserverEntry>>& entries, IntersectionObserver&) final
    {
        ASSERT(!entries.isEmpty());

        for (auto& entry : entries) {
            if (!entry->isIntersecting())
                continue;
            if (RefPtr iframe = dynamicDowncast<HTMLIFrameElement>(entry->target())) {
                iframe->lazyLoadFrameObserver().unobserve();
                iframe->loadDeferredFrame();
            }
        }
        return { };
    }

    CallbackResult<void> handleEventRethrowingException(IntersectionObserver& thisObserver, const Vector<Ref<IntersectionObserverEntry>>& entries, IntersectionObserver& observer) final
    {
        return handleEvent(thisObserver, entries, observer);
    }
};

LazyLoadFrameObserver::LazyLoadFrameObserver(HTMLIFrameElement& element)
    : m_element(element)
{
}

void LazyLoadFrameObserver::observe(const AtomString& frameURL, const ReferrerPolicy& referrerPolicy)
{
    auto& frameObserver = m_element->lazyLoadFrameObserver();
    auto* intersectionObserver = frameObserver.intersectionObserver(m_element->protectedDocument());
    if (!intersectionObserver)
        return;
    m_frameURL = frameURL;
    m_referrerPolicy = referrerPolicy;
    intersectionObserver->observe(m_element);
}

void LazyLoadFrameObserver::unobserve()
{
    auto& frameObserver = m_element->lazyLoadFrameObserver();
    ASSERT(frameObserver.isObserved(m_element));
    frameObserver.m_observer->unobserve(m_element);
}

IntersectionObserver* LazyLoadFrameObserver::intersectionObserver(Document& document)
{
    if (!m_observer) {
        auto callback = LazyFrameLoadIntersectionObserverCallback::create(document);
        IntersectionObserver::Init options { std::nullopt, emptyString(), { } };
        auto observer = IntersectionObserver::create(document, WTFMove(callback), WTFMove(options));
        if (observer.hasException())
            return nullptr;
        m_observer = observer.releaseReturnValue();
    }
    return m_observer.get();
}

bool LazyLoadFrameObserver::isObserved(Element& element) const
{
    return m_observer && m_observer->isObserving(element);
}

}
