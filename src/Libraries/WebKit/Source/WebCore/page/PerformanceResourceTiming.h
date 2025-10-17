/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 11, 2022.
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

#include "NetworkLoadMetrics.h"
#include "PerformanceEntry.h"
#include "ResourceTiming.h"
#include <wtf/Ref.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class PerformanceServerTiming;
class ResourceTiming;

class PerformanceResourceTiming : public PerformanceEntry {
public:
    static Ref<PerformanceResourceTiming> create(MonotonicTime timeOrigin, ResourceTiming&&);

    const String& initiatorType() const { return m_resourceTiming.initiatorType(); }
    const String& nextHopProtocol() const;

    double workerStart() const;
    double redirectStart() const;
    double redirectEnd() const;
    double fetchStart() const;
    double domainLookupStart() const;
    double domainLookupEnd() const;
    double connectStart() const;
    double connectEnd() const;
    double secureConnectionStart() const;
    double requestStart() const;
    double responseStart() const;
    double responseEnd() const;
    uint64_t transferSize() const;
    uint64_t encodedBodySize() const;
    uint64_t decodedBodySize() const;

    const Vector<Ref<PerformanceServerTiming>>& serverTiming() const { return m_serverTiming; }

    Type performanceEntryType() const override { return Type::Resource; }
    ASCIILiteral entryType() const override { return "resource"_s; }

protected:
    PerformanceResourceTiming(MonotonicTime timeOrigin, ResourceTiming&&);
    ~PerformanceResourceTiming();

    bool isLoadedFromServiceWorker() const { return m_resourceTiming.isLoadedFromServiceWorker(); }

    MonotonicTime m_timeOrigin;
    ResourceTiming m_resourceTiming;
    Vector<Ref<PerformanceServerTiming>> m_serverTiming;
};

} // namespace WebCore
