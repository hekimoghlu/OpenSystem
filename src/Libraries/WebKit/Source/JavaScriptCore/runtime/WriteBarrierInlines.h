/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 13, 2025.
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

#include "VM.h"
#include "WriteBarrier.h"

namespace JSC {

template <typename T, typename Traits>
inline void WriteBarrierBase<T, Traits>::set(VM& vm, const JSCell* owner, T* value)
{
    ASSERT(value);
    ASSERT(!Options::useConcurrentJIT() || !isCompilationThread());
    validateCell(value);
    setEarlyValue(vm, owner, value);
}

template <typename T, typename Traits>
inline void WriteBarrierBase<T, Traits>::setMayBeNull(VM& vm, const JSCell* owner, T* value)
{
    if (value)
        validateCell(value);
    setEarlyValue(vm, owner, value);
}

template <typename T, typename Traits>
inline void WriteBarrierBase<T, Traits>::setEarlyValue(VM& vm, const JSCell* owner, T* value)
{
    Traits::exchange(this->m_cell, value);
    vm.writeBarrier(owner, static_cast<JSCell*>(value));
}

inline void WriteBarrierBase<Unknown, RawValueTraits<Unknown>>::set(VM& vm, const JSCell* owner, JSValue value)
{
    ASSERT(!Options::useConcurrentJIT() || !isCompilationThread());
    updateEncodedJSValueConcurrent(m_value, JSValue::encode(value));
    vm.writeBarrier(owner, value);
}

inline void WriteBarrierStructureID::set(VM& vm, const JSCell* owner, Structure* value)
{
    ASSERT(value);
    ASSERT(!Options::useConcurrentJIT() || !isCompilationThread());
    validateCell(reinterpret_cast<JSCell*>(value));
    setEarlyValue(vm, owner, value);
}

inline void WriteBarrierStructureID::setMayBeNull(VM& vm, const JSCell* owner, Structure* value)
{
    if (value)
        validateCell(reinterpret_cast<JSCell*>(value));
    setEarlyValue(vm, owner, value);
}

inline void WriteBarrierStructureID::setEarlyValue(VM& vm, const JSCell* owner, Structure* value)
{
    if (!value) {
        m_structureID = { };
        return;
    }
    m_structureID = StructureID::encode(value);
    vm.writeBarrier(owner, reinterpret_cast<JSCell*>(value));
}

} // namespace JSC 
