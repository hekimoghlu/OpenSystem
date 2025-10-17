/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 2, 2024.
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

#include "ExceptionOr.h"
#include "PlatformTimeRanges.h"

namespace WebCore {

class TimeRanges : public RefCounted<TimeRanges> {
public:
    WEBCORE_EXPORT static Ref<TimeRanges> create();
    WEBCORE_EXPORT static Ref<TimeRanges> create(double start, double end);
    static Ref<TimeRanges> create(const PlatformTimeRanges&);

    WEBCORE_EXPORT ExceptionOr<double> start(unsigned index) const;
    WEBCORE_EXPORT ExceptionOr<double> end(unsigned index) const;

    WEBCORE_EXPORT Ref<TimeRanges> copy() const;
    void invert();
    WEBCORE_EXPORT void intersectWith(const TimeRanges&);
    void unionWith(const TimeRanges&);
    
    WEBCORE_EXPORT unsigned length() const;

    WEBCORE_EXPORT void add(double start, double end, AddTimeRangeOption = AddTimeRangeOption::None);
    bool contain(double time) const;
    
    size_t find(double time) const;
    WEBCORE_EXPORT double nearest(double time) const;
    double totalDuration() const;

    const PlatformTimeRanges& ranges() const { return m_ranges; }
    PlatformTimeRanges& ranges() { return m_ranges; }

private:
    WEBCORE_EXPORT TimeRanges();
    WEBCORE_EXPORT TimeRanges(double start, double end);
    explicit TimeRanges(const PlatformTimeRanges&);

    PlatformTimeRanges m_ranges;
};

} // namespace WebCore
