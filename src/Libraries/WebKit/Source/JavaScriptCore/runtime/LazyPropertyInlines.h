/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 19, 2022.
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

#include "DeferTermination.h"
#include "Heap.h"
#include "VMTraps.h"
#include <wtf/StdLibExtras.h>

namespace JSC {

template<typename OwnerType, typename ElementType>
void LazyProperty<OwnerType, ElementType>::Initializer::set(ElementType* value) const
{
    property.set(vm, owner, value);
}

template<typename OwnerType, typename ElementType>
template<typename Func>
void LazyProperty<OwnerType, ElementType>::initLater(const Func&)
{
    static_assert(isStatelessLambda<Func>());
    // Logically we just want to stuff the function pointer into m_pointer, but then we'd be sad
    // because a function pointer is not guaranteed to be a multiple of anything. The tag bits
    // may be used for things. We address this problem by indirecting through a global const
    // variable. The "theFunc" variable is guaranteed to be native-aligned, i.e. at least a
    // multiple of 4.
    static constexpr FuncType theFunc = &callFunc<Func>;
    m_pointer = lazyTag | std::bit_cast<uintptr_t>(&theFunc);
}

template<typename OwnerType, typename ElementType>
void LazyProperty<OwnerType, ElementType>::setMayBeNull(VM& vm, const OwnerType* owner, ElementType* value)
{
    m_pointer = std::bit_cast<uintptr_t>(value);
    RELEASE_ASSERT(!(m_pointer & lazyTag));
    vm.writeBarrier(owner, value);
}

template<typename OwnerType, typename ElementType>
void LazyProperty<OwnerType, ElementType>::set(VM& vm, const OwnerType* owner, ElementType* value)
{
    RELEASE_ASSERT(value);
    setMayBeNull(vm, owner, value);
}

template<typename OwnerType, typename ElementType>
template<typename Visitor>
void LazyProperty<OwnerType, ElementType>::visit(Visitor& visitor)
{
    if (m_pointer && !(m_pointer & lazyTag))
        visitor.appendUnbarriered(std::bit_cast<ElementType*>(m_pointer));
}

template<typename OwnerType, typename ElementType>
void LazyProperty<OwnerType, ElementType>::dump(PrintStream& out) const
{
    if (!m_pointer) {
        out.print("<null>");
        return;
    }
    if (m_pointer & lazyTag) {
        out.print("Lazy:", RawHex(m_pointer & ~lazyTag));
        if (m_pointer & initializingTag)
            out.print("(Initializing)");
        return;
    }
    out.print(RawHex(m_pointer));
}

template<typename OwnerType, typename ElementType>
template<typename Func>
ElementType* LazyProperty<OwnerType, ElementType>::callFunc(const Initializer& initializer)
{
    if (initializer.property.m_pointer & initializingTag)
        return nullptr;

    DeferTerminationForAWhile deferTerminationForAWhile { initializer.vm };
    initializer.property.m_pointer |= initializingTag;
    callStatelessLambda<void, Func>(initializer);
    RELEASE_ASSERT(!(initializer.property.m_pointer & lazyTag));
    RELEASE_ASSERT(!(initializer.property.m_pointer & initializingTag));
    return std::bit_cast<ElementType*>(initializer.property.m_pointer);
}

} // namespace JSC
