/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 31, 2023.
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

#if ENABLE(DAMAGE_TRACKING)
#include "FloatRect.h"
#include "Region.h"
#include <wtf/ForbidHeapAllocation.h>

namespace WebCore {

class Damage {
    WTF_FORBID_HEAP_ALLOCATION;

public:
    using Rects = Vector<IntRect, 1>;

    enum class Propagation : uint8_t {
        None,
        Region,
        Unified,
    };

    Damage() = default;
    Damage(Damage&&) = default;
    Damage(const Damage&) = default;
    Damage& operator=(const Damage&) = default;
    Damage& operator=(Damage&&) = default;

    static const Damage& invalid()
    {
        static const Damage invalidDamage(true);
        return invalidDamage;
    }

    ALWAYS_INLINE const Region& region() const { return m_region; }
    ALWAYS_INLINE IntRect bounds() const { return m_region.bounds(); }
    ALWAYS_INLINE Rects rects() const { return m_region.rects(); }
    ALWAYS_INLINE bool isEmpty() const  { return !m_invalid && m_region.isEmpty(); }
    ALWAYS_INLINE bool isInvalid() const { return m_invalid; }

    void invalidate()
    {
        m_invalid = true;
        m_region = Region();
    }

    ALWAYS_INLINE void add(const Region& region)
    {
        if (isInvalid())
            return;
        m_region.unite(region);
        mergeIfNeeded();
    }

    ALWAYS_INLINE void add(const IntRect& rect)
    {
        if (isInvalid())
            return;
        m_region.unite(rect);
        mergeIfNeeded();
    }

    ALWAYS_INLINE void add(const FloatRect& rect)
    {
        add(enclosingIntRect(rect));
    }

    ALWAYS_INLINE void add(const Damage& other)
    {
        m_invalid = other.isInvalid();
        add(other.m_region);
    }

private:
    bool m_invalid { false };
    Region m_region;

    // From RenderView.cpp::repaintViewRectangle():
    // Region will get slow if it gets too complex.
    // Merge all rects so far to bounds if this happens.
    static constexpr auto maximumGridSize = 16 * 16;

    ALWAYS_INLINE void mergeIfNeeded()
    {
        if (UNLIKELY(m_region.gridSize() > maximumGridSize))
            m_region = Region(m_region.bounds());
    }

    explicit Damage(bool invalid)
        : m_invalid(invalid)
    {
    }

    friend struct IPC::ArgumentCoder<Damage, void>;

    friend bool operator==(const Damage&, const Damage&) = default;
};

static inline WTF::TextStream& operator<<(WTF::TextStream& ts, const Damage& damage)
{
    if (damage.isInvalid())
        return ts << "Damage[invalid]";
    return ts << "Damage" << damage.rects();
}

};

#endif // ENABLE(DAMAGE_TRACKING)
