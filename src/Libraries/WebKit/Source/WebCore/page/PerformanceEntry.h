/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 10, 2024.
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

#include "Performance.h"
#include <wtf/RefCounted.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

DECLARE_ALLOCATOR_WITH_HEAP_IDENTIFIER(PerformanceEntry);
class PerformanceEntry : public RefCounted<PerformanceEntry> {
    WTF_MAKE_FAST_ALLOCATED_WITH_HEAP_IDENTIFIER(PerformanceEntry);
public:
    virtual ~PerformanceEntry();

    const String& name() const { return m_name; }
    virtual double startTime() const { return m_startTime; }
    virtual double duration() const { return m_duration; }

    enum class Type : uint8_t {
        Navigation  = 1 << 0,
        Mark        = 1 << 1,
        Measure     = 1 << 2,
        Resource    = 1 << 3,
        Paint       = 1 << 4
    };

    virtual Type performanceEntryType() const = 0;
    virtual ASCIILiteral entryType() const = 0;

    static std::optional<Type> parseEntryTypeString(const String& entryType);

    static bool startTimeCompareLessThan(const Ref<PerformanceEntry>& a, const Ref<PerformanceEntry>& b)
    {
        return a->startTime() < b->startTime();
    }

protected:
    PerformanceEntry(const String& name, double startTime, double finishTime);

private:
    const String m_name;
    const double m_startTime;
    const double m_duration;
};

} // namespace WebCore
