/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 9, 2025.
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

#include "GradientColorStop.h"
#include <algorithm>
#include <optional>
#include <wtf/Forward.h>
#include <wtf/Vector.h>

namespace WebCore {

class GradientColorStops {
public:
    using StopVector = Vector<GradientColorStop, 2>;

    struct Sorted {
        StopVector stops;
    };

    GradientColorStops()
        : m_isSorted { true }
    {
    }

    GradientColorStops(StopVector stops)
        : m_stops { WTFMove(stops) }
        , m_isSorted { false }
    {
    }

    GradientColorStops(Sorted sortedStops)
        : m_stops { WTFMove(sortedStops.stops) }
        , m_isSorted { true }
    {
        ASSERT(validateIsSorted());
    }

    void addColorStop(GradientColorStop stop)
    {
        if (!m_stops.isEmpty() && m_stops.last().offset > stop.offset)
            m_isSorted = false;
        m_stops.append(WTFMove(stop));
    }

    void sort()
    {
        if (m_isSorted)
            return;

        std::stable_sort(m_stops.begin(), m_stops.end(), [] (auto& a, auto& b) {
            return a.offset < b.offset;
        });
        m_isSorted = true;
    }

    const GradientColorStops& sorted() const
    {
        const_cast<GradientColorStops*>(this)->sort();
        return *this;
    }

    size_t size() const { return m_stops.size(); }
    bool isEmpty() const { return m_stops.isEmpty(); }

    StopVector::const_iterator begin() const { return m_stops.begin(); }
    StopVector::const_iterator end() const { return m_stops.end(); }

    template<typename MapFunction> GradientColorStops mapColors(MapFunction&& mapFunction) const
    {
        return {
            m_stops.map<StopVector>([&] (const GradientColorStop& stop) -> GradientColorStop {
                return { stop.offset, mapFunction(stop.color) };
            }),
            m_isSorted
        };
    }

    const StopVector& stops() const { return m_stops; }

private:
    GradientColorStops(StopVector stops, bool isSorted)
        : m_stops { WTFMove(stops) }
        , m_isSorted { isSorted }
    {
    }

#if ASSERT_ENABLED
    bool validateIsSorted() const
    {
        return std::is_sorted(m_stops.begin(), m_stops.end(), [] (auto& a, auto& b) {
            return a.offset < b.offset;
        });
    }
#endif

    StopVector m_stops;
    bool m_isSorted;
};

TextStream& operator<<(TextStream&, const GradientColorStops&);

} // namespace WebCore
