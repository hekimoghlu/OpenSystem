/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 27, 2024.
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

#ifndef NDEBUG
#include <wtf/text/TextStream.h>
#endif

namespace WebCore {

// Class template representing a closed interval which can hold arbitrary
// endpoints and a piece of user data. Ordering and equality are defined
// including the UserData, except in the special case of WeakPtr.
//
// Both the T and UserData types must support a copy constructor, operator<,
// operator==, and operator=, except that this does not depend on operator<
// or operator== for WeakPtr.
//
// In debug mode, printing of intervals and the data they contain is
// enabled. This uses WTF::TextStream.
//
// Note that this class supplies a copy constructor and assignment
// operator in order so it can be stored in the red-black tree.

// FIXME: The prefix "POD" here isn't correct; this works with non-POD types.

template<class T, class UserData> class PODIntervalBase {
public:
    const T& low() const { return m_low; }
    const T& high() const { return m_high; }
    const UserData& data() const { return m_data; }

    bool overlaps(const T& low, const T& high) const
    {
        return !(m_high < low || high < m_low);
    }

    bool overlaps(const PODIntervalBase& other) const
    {
        return overlaps(other.m_low, other.m_high);
    }

    const T& maxHigh() const { return m_maxHigh; }
    void setMaxHigh(const T& maxHigh) { m_maxHigh = maxHigh; }

protected:
    PODIntervalBase(const T& low, const T& high, UserData&& data)
        : m_low(low)
        , m_high(high)
        , m_data(WTFMove(data))
        , m_maxHigh(high)
    {
    }

private:
    T m_low;
    T m_high;
    UserData m_data { };
    T m_maxHigh;
};

template<class T, class UserData> class PODInterval : public PODIntervalBase<T, UserData> {
public:
    PODInterval(const T& low, const T& high, UserData&& data = { })
        : PODIntervalBase<T, UserData>(low, high, WTFMove(data))
    {
    }

    PODInterval(const T& low, const T& high, const UserData& data)
        : PODIntervalBase<T, UserData>(low, high, UserData { data })
    {
    }

    bool operator<(const PODInterval& other) const
    {
        if (Base::low() < other.Base::low())
            return true;
        if (other.Base::low() < Base::low())
            return false;
        if (Base::high() < other.Base::high())
            return true;
        if (other.Base::high() < Base::high())
            return false;
        return Base::data() < other.Base::data();
    }

    bool operator==(const PODInterval& other) const
    {
        return Base::low() == other.Base::low()
            && Base::high() == other.Base::high()
            && Base::data() == other.Base::data();
    }

private:
    using Base = PODIntervalBase<T, UserData>;
};

template<typename T, typename U, typename WeakPtrImpl> class PODInterval<T, WeakPtr<U, WeakPtrImpl>> : public PODIntervalBase<T, WeakPtr<U, WeakPtrImpl>> {
public:
    PODInterval(const T& low, const T& high, WeakPtr<U, WeakPtrImpl>&& data)
        : PODIntervalBase<T, WeakPtr<U, WeakPtrImpl>>(low, high, WTFMove(data))
    {
    }

    bool operator<(const PODInterval& other) const
    {
        if (Base::low() < other.Base::low())
            return true;
        if (other.Base::low() < Base::low())
            return false;
        return Base::high() < other.Base::high();
    }

private:
    using Base = PODIntervalBase<T, WeakPtr<U, WeakPtrImpl>>;
};

#ifndef NDEBUG

template<class T, class UserData>
TextStream& operator<<(TextStream& stream, const PODInterval<T, UserData>& interval)
{
    return stream << "[PODInterval (" << interval.low() << ", " << interval.high() << "), data=" << *interval.data() << ", maxHigh=" << interval.maxHigh() << ']';
}

#endif

} // namespace WebCore
