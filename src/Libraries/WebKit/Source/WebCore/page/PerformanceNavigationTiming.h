/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 9, 2021.
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

#include "DocumentEventTiming.h"
#include "DocumentLoadTiming.h"
#include "PerformanceResourceTiming.h"
#include <wtf/MonotonicTime.h>
#include <wtf/RefCounted.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class CachedResource;
class NetworkLoadMetrics;
class SecurityOrigin;
enum class NavigationType : uint8_t;

class PerformanceNavigationTiming final : public PerformanceResourceTiming {
public:
    enum class NavigationType : uint8_t {
        Navigate,
        Reload,
        Back_forward,
        Prerender,
    };

    template<typename... Args> static Ref<PerformanceNavigationTiming> create(Args&&... args) { return adoptRef(*new PerformanceNavigationTiming(std::forward<Args>(args)...)); }
    ~PerformanceNavigationTiming();

    Type performanceEntryType() const final { return Type::Navigation; }
    ASCIILiteral entryType() const final { return "navigation"_s; }

    double unloadEventStart() const;
    double unloadEventEnd() const;
    double domInteractive() const;
    double domContentLoadedEventStart() const;
    double domContentLoadedEventEnd() const;
    double domComplete() const;
    double loadEventStart() const;
    double loadEventEnd() const;
    NavigationType type() const;
    unsigned short redirectCount() const;

    double startTime() const final;
    double duration() const final;

    DocumentEventTiming& documentEventTiming() { return m_documentEventTiming; }
    DocumentLoadTiming& documentLoadTiming() { return m_documentLoadTiming; }
    void navigationFinished(const NetworkLoadMetrics&);

private:
    PerformanceNavigationTiming(MonotonicTime timeOrigin, CachedResource&, const DocumentLoadTiming&, const NetworkLoadMetrics&, const DocumentEventTiming&, const SecurityOrigin&, WebCore::NavigationType);

    double millisecondsSinceOrigin(MonotonicTime) const;
    bool sameOriginCheckFails() const;

    DocumentEventTiming m_documentEventTiming;
    DocumentLoadTiming m_documentLoadTiming;
    NavigationType m_navigationType;
};

} // namespace WebCore
