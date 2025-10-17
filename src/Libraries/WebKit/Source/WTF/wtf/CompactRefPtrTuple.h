/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 5, 2022.
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

#include <wtf/CompactPointerTuple.h>
#include <wtf/Noncopyable.h>
#include <wtf/RefPtr.h>

namespace WTF {

template<typename T, typename Type>
class CompactRefPtrTuple final {
    WTF_MAKE_FAST_ALLOCATED;
    WTF_MAKE_NONCOPYABLE(CompactRefPtrTuple);

    static_assert(::allowCompactPointers<T>());
public:
    CompactRefPtrTuple() = default;
    ~CompactRefPtrTuple()
    {
        WTF::DefaultRefDerefTraits<T>::derefIfNotNull(m_data.pointer());
    }

    T* pointer() const
    {
        return m_data.pointer();
    }

    void setPointer(T* pointer)
    {
        auto* old = m_data.pointer();
        m_data.setPointer(WTF::DefaultRefDerefTraits<T>::refIfNotNull(pointer));
        WTF::DefaultRefDerefTraits<T>::derefIfNotNull(old);
    }

    void setPointer(RefPtr<T>&& pointer)
    {
        auto willRelease = WTFMove(pointer);
        auto* old = m_data.pointer();
        m_data.setPointer(willRelease.leakRef());
        WTF::DefaultRefDerefTraits<T>::derefIfNotNull(old);
    }

    void setPointer(Ref<T>&& pointer)
    {
        auto willRelease = WTFMove(pointer);
        auto* old = m_data.pointer();
        m_data.setPointer(&willRelease.leakRef());
        WTF::DefaultRefDerefTraits<T>::derefIfNotNull(old);
    }

    Type type() const { return m_data.type(); }
    void setType(Type type)
    {
        m_data.setType(type);
    }

private:
    CompactPointerTuple<T*, Type> m_data;
};

} // namespace WTF

using WTF::CompactRefPtrTuple;
