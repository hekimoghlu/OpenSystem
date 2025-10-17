/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 15, 2025.
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
#include "LazyLoadImageObserver.h"

#include "DocumentInlines.h"
#include "HTMLImageElement.h"
#include "IntersectionObserverCallback.h"
#include "IntersectionObserverEntry.h"
#include "LocalFrame.h"
#include <limits>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(LazyLoadImageObserver);

class LazyImageLoadIntersectionObserverCallback final : public IntersectionObserverCallback {
public:
    static Ref<LazyImageLoadIntersectionObserverCallback> create(Document& document)
    {
        return adoptRef(*new LazyImageLoadIntersectionObserverCallback(document));
    }

private:
    LazyImageLoadIntersectionObserverCallback(Document& document)
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
            if (RefPtr element = dynamicDowncast<HTMLImageElement>(entry->target())) {
                element->loadDeferredImage();
                element->document().lazyLoadImageObserver().unobserve(*element, element->document());
            }
        }
        return { };
    }

    CallbackResult<void> handleEventRethrowingException(IntersectionObserver& thisObserver, const Vector<Ref<IntersectionObserverEntry>>& entries, IntersectionObserver& observer) final
    {
        return handleEvent(thisObserver, entries, observer);
    }
};

void LazyLoadImageObserver::observe(Element& element)
{
    auto& observer = element.document().lazyLoadImageObserver();
    auto* intersectionObserver = observer.intersectionObserver(element.protectedDocument());
    if (!intersectionObserver)
        return;
    intersectionObserver->observe(element);
}

void LazyLoadImageObserver::unobserve(Element& element, Document& document)
{
    if (auto& observer = document.lazyLoadImageObserver().m_observer)
        observer->unobserve(element);
}

IntersectionObserver* LazyLoadImageObserver::intersectionObserver(Document& document)
{
    if (!m_observer) {
        auto callback = LazyImageLoadIntersectionObserverCallback::create(document);
        static NeverDestroyed<const String> lazyLoadingRootMarginFallback(MAKE_STATIC_STRING_IMPL("100%"));
        IntersectionObserver::Init options { &document, lazyLoadingRootMarginFallback, { } };
        auto observer = IntersectionObserver::create(document, WTFMove(callback), WTFMove(options));
        if (observer.hasException())
            return nullptr;
        m_observer = observer.returnValue().ptr();
    }
    return m_observer.get();
}

bool LazyLoadImageObserver::isObserved(Element& element) const
{
    return m_observer && m_observer->isObserving(element);
}

}
