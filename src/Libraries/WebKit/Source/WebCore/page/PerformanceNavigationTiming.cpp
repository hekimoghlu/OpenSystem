/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 15, 2022.
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
#include "PerformanceNavigationTiming.h"

#include "CachedResource.h"
#include "ResourceTiming.h"

namespace WebCore {

static PerformanceNavigationTiming::NavigationType toPerformanceNavigationTimingNavigationType(NavigationType navigationType)
{
    switch (navigationType) {
    case NavigationType::BackForward:
        return PerformanceNavigationTiming::NavigationType::Back_forward;
    case NavigationType::Reload:
        return PerformanceNavigationTiming::NavigationType::Reload;
    case NavigationType::LinkClicked:
    case NavigationType::FormSubmitted:
    case NavigationType::FormResubmitted:
    case NavigationType::Other:
        return PerformanceNavigationTiming::NavigationType::Navigate;
    }
    ASSERT_NOT_REACHED();
    return PerformanceNavigationTiming::NavigationType::Navigate;
}

PerformanceNavigationTiming::PerformanceNavigationTiming(MonotonicTime timeOrigin, CachedResource& resource, const DocumentLoadTiming& documentLoadTiming, const NetworkLoadMetrics& metrics, const DocumentEventTiming& documentEventTiming, const SecurityOrigin& origin, WebCore::NavigationType navigationType)
    : PerformanceResourceTiming(timeOrigin, ResourceTiming::fromLoad(resource, resource.response().url(), "navigation"_s, documentLoadTiming, metrics, origin))
    , m_documentEventTiming(documentEventTiming)
    , m_documentLoadTiming(documentLoadTiming)
    , m_navigationType(toPerformanceNavigationTimingNavigationType(navigationType)) { }

PerformanceNavigationTiming::~PerformanceNavigationTiming() = default;

double PerformanceNavigationTiming::millisecondsSinceOrigin(MonotonicTime time) const
{
    if (!time)
        return 0;
    return Performance::reduceTimeResolution(time - m_timeOrigin).milliseconds();
}

bool PerformanceNavigationTiming::sameOriginCheckFails() const
{
    // https://www.w3.org/TR/navigation-timing-2/#dfn-same-origin-check
    return m_resourceTiming.networkLoadMetrics().hasCrossOriginRedirect
        || !m_documentLoadTiming.hasSameOriginAsPreviousDocument();
}

double PerformanceNavigationTiming::unloadEventStart() const
{
    if (sameOriginCheckFails())
        return 0.0;
    return millisecondsSinceOrigin(m_documentLoadTiming.unloadEventStart());
}

double PerformanceNavigationTiming::unloadEventEnd() const
{
    if (sameOriginCheckFails())
        return 0.0;
    return millisecondsSinceOrigin(m_documentLoadTiming.unloadEventEnd());
}

double PerformanceNavigationTiming::domInteractive() const
{
    return millisecondsSinceOrigin(m_documentEventTiming.domInteractive);
}

double PerformanceNavigationTiming::domContentLoadedEventStart() const
{
    return millisecondsSinceOrigin(m_documentEventTiming.domContentLoadedEventStart);
}

double PerformanceNavigationTiming::domContentLoadedEventEnd() const
{
    return millisecondsSinceOrigin(m_documentEventTiming.domContentLoadedEventEnd);
}

double PerformanceNavigationTiming::domComplete() const
{
    return millisecondsSinceOrigin(m_documentEventTiming.domComplete);
}

double PerformanceNavigationTiming::loadEventStart() const
{
    return millisecondsSinceOrigin(m_documentLoadTiming.loadEventStart());
}

double PerformanceNavigationTiming::loadEventEnd() const
{
    return millisecondsSinceOrigin(m_documentLoadTiming.loadEventEnd());
}

PerformanceNavigationTiming::NavigationType PerformanceNavigationTiming::type() const
{
    return m_navigationType;
}

unsigned short PerformanceNavigationTiming::redirectCount() const
{
    if (m_resourceTiming.networkLoadMetrics().hasCrossOriginRedirect)
        return 0;

    return m_resourceTiming.networkLoadMetrics().redirectCount;
}

double PerformanceNavigationTiming::startTime() const
{
    // https://www.w3.org/TR/navigation-timing-2/#dom-PerformanceNavigationTiming-startTime
    return 0.0;
}

double PerformanceNavigationTiming::duration() const
{
    // https://www.w3.org/TR/navigation-timing-2/#dom-PerformanceNavigationTiming-duration
    return loadEventEnd() - startTime();
}

void PerformanceNavigationTiming::navigationFinished(const NetworkLoadMetrics& metrics)
{
    m_documentLoadTiming.markEndTime();
    m_resourceTiming.networkLoadMetrics().updateFromFinalMetrics(metrics);
}

} // namespace WebCore
