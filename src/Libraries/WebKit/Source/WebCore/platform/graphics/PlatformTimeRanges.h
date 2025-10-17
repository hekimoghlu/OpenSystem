/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 16, 2024.
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

#include <algorithm>
#include <wtf/ArgumentCoder.h>
#include <wtf/MediaTime.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>

OBJC_CLASS NSArray;

namespace WTF {
class PrintStream;
}

namespace WebCore {

enum class AddTimeRangeOption : uint8_t {
    None,
    EliminateSmallGaps,
};

class WEBCORE_EXPORT PlatformTimeRanges final {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(PlatformTimeRanges, WEBCORE_EXPORT);
public:
    PlatformTimeRanges();
    PlatformTimeRanges(const MediaTime& start, const MediaTime& end);

    PlatformTimeRanges copyWithEpsilon(const MediaTime&) const;

    static const PlatformTimeRanges& emptyRanges();
    static MediaTime timeFudgeFactor();

    MediaTime start(unsigned index) const;
    MediaTime start(unsigned index, bool& valid) const;
    MediaTime end(unsigned index) const;
    MediaTime end(unsigned index, bool& valid) const;
    MediaTime duration(unsigned index) const;
    MediaTime maximumBufferedTime() const;
    MediaTime minimumBufferedTime() const;

    void invert();
    void intersectWith(const PlatformTimeRanges&);
    void unionWith(const PlatformTimeRanges&);
    PlatformTimeRanges& operator+=(const PlatformTimeRanges&);
    PlatformTimeRanges& operator-=(const PlatformTimeRanges&);

    unsigned length() const { return m_ranges.size(); }

    void add(const MediaTime& start, const MediaTime& end, AddTimeRangeOption = AddTimeRangeOption::None);
    void clear();
    
    bool contain(const MediaTime&) const;

    size_t find(const MediaTime&) const;
    size_t findWithEpsilon(const MediaTime&, const MediaTime& epsilon);
    
    MediaTime nearest(const MediaTime&) const;

    MediaTime totalDuration() const;

    void dump(PrintStream&) const;
    String toString() const;

    // We consider all the Ranges to be semi-bounded as follow: [start, end[
    struct Range {
        MediaTime start;
        MediaTime end;

        inline bool isEmpty() const
        {
            return start == end;
        }

        inline bool isPointInRange(const MediaTime& point) const
        {
            return start <= point && point < end;
        }
        
        inline bool isOverlappingRange(const Range& range) const
        {
            return isPointInRange(range.start) || isPointInRange(range.end) || range.isPointInRange(start);
        }

        inline bool isContiguousWithRange(const Range& range) const
        {
            return range.start == end || range.end == start;
        }
        
        inline Range unionWithOverlappingOrContiguousRange(const Range& range) const
        {
            Range ret;

            ret.start = std::min(start, range.start);
            ret.end = std::max(end, range.end);

            return ret;
        }

        inline bool isBeforeRange(const Range& range) const
        {
            return range.start >= end;
        }

        friend bool operator==(const Range&, const Range&) = default;
    };

    friend bool operator==(const PlatformTimeRanges&, const PlatformTimeRanges&) = default;

private:
    friend struct IPC::ArgumentCoder<PlatformTimeRanges, void>;

    PlatformTimeRanges(Vector<Range>&&);
    PlatformTimeRanges& operator-=(const Range&);

    size_t findLastRangeIndexBefore(const MediaTime& start, const MediaTime& end) const;

    Vector<Range> m_ranges;
};

#if PLATFORM(COCOA)
RetainPtr<NSArray> makeNSArray(const PlatformTimeRanges&);
#endif

} // namespace WebCore

namespace WTF {
template<typename> struct LogArgument;

template<> struct LogArgument<WebCore::PlatformTimeRanges> {
    static String toString(const WebCore::PlatformTimeRanges& platformTimeRanges) { return platformTimeRanges.toString(); }
};

} // namespace WTF
