/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 23, 2025.
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

#include "LocalDOMWindowProperty.h"
#include <wtf/MonotonicTime.h>
#include <wtf/Ref.h>
#include <wtf/RefCounted.h>

namespace WebCore {

class DocumentLoader;
class DocumentLoadTiming;
class LoadTiming;
class NetworkLoadMetrics;

struct DocumentEventTiming;

class PerformanceTiming : public RefCounted<PerformanceTiming>, public LocalDOMWindowProperty {
public:
    static Ref<PerformanceTiming> create(LocalDOMWindow* window) { return adoptRef(*new PerformanceTiming(window)); }

    unsigned long long navigationStart() const;
    unsigned long long unloadEventStart() const;
    unsigned long long unloadEventEnd() const;
    unsigned long long redirectStart() const;
    unsigned long long redirectEnd() const;
    unsigned long long fetchStart() const;
    unsigned long long domainLookupStart() const;
    unsigned long long domainLookupEnd() const;
    unsigned long long connectStart() const;
    unsigned long long connectEnd() const;
    unsigned long long secureConnectionStart() const;
    unsigned long long requestStart() const;
    unsigned long long responseStart() const;
    unsigned long long responseEnd() const;
    unsigned long long domLoading() const;
    unsigned long long domInteractive() const;
    unsigned long long domContentLoadedEventStart() const;
    unsigned long long domContentLoadedEventEnd() const;
    unsigned long long domComplete() const;
    unsigned long long loadEventStart() const;
    unsigned long long loadEventEnd() const;

private:
    explicit PerformanceTiming(LocalDOMWindow*);

    const DocumentEventTiming* documentEventTiming() const;
    const DocumentLoader* documentLoader() const;
    const DocumentLoadTiming* documentLoadTiming() const;
    const NetworkLoadMetrics* networkLoadMetrics() const;
    unsigned long long monotonicTimeToIntegerMilliseconds(MonotonicTime) const;

    mutable unsigned long long m_navigationStart { 0 };
    mutable unsigned long long m_unloadEventStart { 0 };
    mutable unsigned long long m_unloadEventEnd { 0 };
    mutable unsigned long long m_redirectStart { 0 };
    mutable unsigned long long m_redirectEnd { 0 };
    mutable unsigned long long m_fetchStart { 0 };
    mutable unsigned long long m_domainLookupStart { 0 };
    mutable unsigned long long m_domainLookupEnd { 0 };
    mutable unsigned long long m_connectStart { 0 };
    mutable unsigned long long m_connectEnd { 0 };
    mutable unsigned long long m_secureConnectionStart { 0 };
    mutable unsigned long long m_requestStart { 0 };
    mutable unsigned long long m_responseStart { 0 };
    mutable unsigned long long m_responseEnd { 0 };
    mutable unsigned long long m_domLoading { 0 };
    mutable unsigned long long m_domInteractive { 0 };
    mutable unsigned long long m_domContentLoadedEventStart { 0 };
    mutable unsigned long long m_domContentLoadedEventEnd { 0 };
    mutable unsigned long long m_domComplete { 0 };
    mutable unsigned long long m_loadEventStart { 0 };
    mutable unsigned long long m_loadEventEnd { 0 };
};

} // namespace WebCore
