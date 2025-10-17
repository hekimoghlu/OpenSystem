/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 12, 2023.
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

#include "Length.h"
#include "WebAnimationTypes.h"

namespace WebCore {

namespace Style {
class BuilderState;
}

class Element;

struct SingleTimelineRange {
    enum class Name { Normal, Omitted, Cover, Contain, Entry, Exit, EntryCrossing, ExitCrossing };

    Name name { Name::Normal };
    Length offset;

    bool operator==(const SingleTimelineRange& other) const = default;

    enum class Type : bool { Start, End };
    static bool isDefault(const Length&, Type);
    static bool isDefault(const CSSPrimitiveValue&, Type);
    static Length defaultValue(Type);
    static Length lengthForCSSValue(RefPtr<const CSSPrimitiveValue>, RefPtr<Element>);

    static bool isOffsetValue(const CSSPrimitiveValue&);

    static Name timelineName(CSSValueID);
    static CSSValueID valueID(Name);

    static SingleTimelineRange range(const CSSValue&, Type, Style::BuilderState* = nullptr, RefPtr<Element> = nullptr);
    static RefPtr<CSSValue> parse(TimelineRangeValue&&, RefPtr<Element>, Type);
    TimelineRangeValue serialize() const;
};

struct TimelineRange {
    SingleTimelineRange start;
    SingleTimelineRange end;

    bool operator==(const TimelineRange& other) const = default;

    static TimelineRange defaultForScrollTimeline();
    static TimelineRange defaultForViewTimeline();
    bool isDefault() const { return start.name == SingleTimelineRange::Name::Normal && end.name == SingleTimelineRange::Name::Normal; }
};

WTF::TextStream& operator<<(WTF::TextStream&, const SingleTimelineRange&);

} // namespace WebCore
