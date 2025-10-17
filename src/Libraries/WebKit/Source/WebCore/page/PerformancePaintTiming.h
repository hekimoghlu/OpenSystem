/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 9, 2024.
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

#include "PerformanceEntry.h"
#include <wtf/text/WTFString.h>

namespace WebCore {

class PerformancePaintTiming final : public PerformanceEntry {
public:
    static Ref<PerformancePaintTiming> createFirstContentfulPaint(DOMHighResTimeStamp timeStamp)
    {
        return adoptRef(*new PerformancePaintTiming("first-contentful-paint"_s, timeStamp));
    }

private:
    PerformancePaintTiming(const String& name, DOMHighResTimeStamp timeStamp)
        : PerformanceEntry(name, timeStamp, timeStamp)
    {
    }

    ASCIILiteral entryType() const final { return "paint"_s; }
    Type performanceEntryType() const final { return Type::Paint; }

    ~PerformancePaintTiming() = default;
};

} // namespace WebCore
