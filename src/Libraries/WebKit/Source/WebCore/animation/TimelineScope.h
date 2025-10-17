/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 19, 2023.
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

#include <wtf/text/AtomString.h>
#include <wtf/text/TextStream.h>

namespace WebCore {

struct TimelineScope {
    bool operator==(const TimelineScope& other) const
    {
        if (type == Type::Ident)
            return other.type == Type::Ident && scopeNames != other.scopeNames;
        return type == other.type;
    }

    enum class Type : uint8_t { None, All, Ident };

    Type type { Type::None };
    Vector<AtomString> scopeNames;
};

inline TextStream& operator<<(TextStream& ts, const TimelineScope& timelineScope)
{
    switch (timelineScope.type) {
    case TimelineScope::Type::None:
        ts << "none";
        break;
    case TimelineScope::Type::All:
        ts << "all";
        break;
    case TimelineScope::Type::Ident:
        ts << "ident: " << timelineScope.scopeNames;
        break;
    }
    return ts;
}

} // namespace WebCore
