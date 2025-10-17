/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 5, 2024.
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

struct PositionTryFallback {
    enum class Tactic : uint8_t {
        FlipBlock,
        FlipInline,
        FlipStart
    };
    Vector<Tactic> tactics;

    bool operator==(const PositionTryFallback&) const = default;
};

inline TextStream& operator<<(TextStream& ts, const PositionTryFallback& positionTryFallback)
{
    auto separator = ""_s;
    for (auto& tactic : positionTryFallback.tactics) {
        ts << std::exchange(separator, " "_s);
        switch (tactic) {
        case PositionTryFallback::Tactic::FlipBlock:
            ts << "flip-block";
            break;
        case PositionTryFallback::Tactic::FlipInline:
            ts << "flip-inline";
            break;
        case PositionTryFallback::Tactic::FlipStart:
            ts << "flip-start";
            break;
        }
    }
    return ts;
}

inline TextStream& operator<<(TextStream& ts, const Vector<PositionTryFallback>& positionTryFallbacks)
{
    if (positionTryFallbacks.isEmpty()) {
        ts << "none";
        return ts;
    }
    auto separator = ""_s;
    for (auto& item : positionTryFallbacks) {
        ts << std::exchange(separator, ", "_s);
        ts << item;
    }
    return ts;
}

} // namespace WebCore

