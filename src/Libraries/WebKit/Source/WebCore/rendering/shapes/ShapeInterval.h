/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 24, 2023.
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

#include <wtf/TZoneMallocInlines.h>
#include <wtf/Vector.h>

namespace WebCore {

template <typename T>
class ShapeInterval {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(ShapeInterval);
public:
    ShapeInterval()
        : m_x1(-1)
        , m_x2(-2)
    {
        // The initial values of m_x1,x2 don't matter (unless you're looking
        // at them in the debugger) so long as isUndefined() is true.
        ASSERT(isUndefined());
    }

    ShapeInterval(T x1, T x2)
        : m_x1(x1)
        , m_x2(x2)
    {
        ASSERT(x2 >= x1);
    }

    bool isUndefined() const { return m_x2 < m_x1; }
    T x1() const { return isUndefined() ? 0 : m_x1; }
    T x2() const { return isUndefined() ? 0 : m_x2; }
    T width() const { return isUndefined() ? 0 : m_x2 - m_x1; }
    bool isEmpty() const { return isUndefined() ? true : m_x1 == m_x2; }

    void set(T x1, T x2)
    {
        ASSERT(x2 >= x1);
        m_x1 = x1;
        m_x2 = x2;
    }

    bool overlaps(const ShapeInterval<T>& interval) const
    {
        if (isUndefined() || interval.isUndefined())
            return false;
        return x2() >= interval.x1() && x1() <= interval.x2();
    }

    bool contains(const ShapeInterval<T>& interval) const
    {
        if (isUndefined() || interval.isUndefined())
            return false;
        return x1() <= interval.x1() && x2() >= interval.x2();
    }

    void unite(const ShapeInterval<T>& interval)
    {
        if (interval.isUndefined())
            return;
        if (isUndefined())
            set(interval.x1(), interval.x2());
        else
            set(std::min<T>(x1(), interval.x1()), std::max<T>(x2(), interval.x2()));
    }

    friend bool operator==(const ShapeInterval<T>&, const ShapeInterval<T>&) = default;

private:
    T m_x1;
    T m_x2;
};

typedef ShapeInterval<int> IntShapeInterval;
typedef ShapeInterval<float> FloatShapeInterval;

typedef Vector<IntShapeInterval> IntShapeIntervals;
typedef Vector<FloatShapeInterval> FloatShapeIntervals;

} // namespace WebCore
