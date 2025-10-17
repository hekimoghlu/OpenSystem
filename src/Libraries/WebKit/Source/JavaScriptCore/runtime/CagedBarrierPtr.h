/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 22, 2025.
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

#include "AuxiliaryBarrier.h"
#include <wtf/CagedPtr.h>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace JSC {

class JSCell;
class VM;

// This is a convenient combo of AuxiliaryBarrier and CagedPtr.

template<Gigacage::Kind passedKind, typename T>
class CagedBarrierPtr {
public:
    static constexpr Gigacage::Kind kind = passedKind;
    using Type = T;
    using CagedType = CagedPtr<kind, Type>;
    
    CagedBarrierPtr() = default;
    
    template<typename U>
    CagedBarrierPtr(VM& vm, JSCell* cell, U&& value)
    {
        m_barrier.set(vm, cell, CagedType(std::forward<U>(value)));
    }
    
    void clear() { m_barrier.clear(); }
    
    template<typename U>
    void set(VM& vm, JSCell* cell, U&& value)
    {
        m_barrier.set(vm, cell, CagedType(std::forward<U>(value)));
    }
    
    T* get() const { return m_barrier.get().get(); }
    T* getMayBeNull() const { return m_barrier.get().getMayBeNull(); }
    T* getUnsafe() const { return m_barrier.get().getUnsafe(); }

    // We need the template here so that the type of U is deduced at usage time rather than class time. U should always be T.
    template<typename U = T>
    typename std::enable_if<!std::is_same<void, U>::value, T>::type&
    /* T& */ at(size_t index) const { return get()[index]; }

    bool operator==(const CagedBarrierPtr& other) const
    {
        return m_barrier.get() == other.m_barrier.get();
    }
    
    explicit operator bool() const
    {
        return !!m_barrier.get();
    }
    
    template<typename U>
    void setWithoutBarrier(U&& value) { m_barrier.setWithoutBarrier(CagedType(std::forward<U>(value))); }

    T* rawBits() const
    {
        return m_barrier.get().rawBits();
    }
    
private:
    AuxiliaryBarrier<CagedType> m_barrier;
};

} // namespace JSC

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
