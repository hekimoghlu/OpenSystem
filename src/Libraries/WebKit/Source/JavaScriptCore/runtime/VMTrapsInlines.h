/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 10, 2023.
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

namespace JSC {

ALWAYS_INLINE VM& VMTraps::vm() const
{
    return *std::bit_cast<VM*>(std::bit_cast<uintptr_t>(this) - OBJECT_OFFSETOF(VM, m_traps));
}

inline void VMTraps::deferTermination(DeferAction deferAction)
{
    auto originalCount = m_deferTerminationCount++;
    ASSERT(m_deferTerminationCount < UINT_MAX);
    if (UNLIKELY(originalCount == 0 && vm().exception()))
        deferTerminationSlow(deferAction);
}

inline void VMTraps::undoDeferTermination(DeferAction deferAction)
{
    ASSERT(m_deferTerminationCount > 0);
    ASSERT(!m_suspendedTerminationException || vm().hasTerminationRequest());
    if (UNLIKELY(--m_deferTerminationCount == 0 && vm().hasTerminationRequest()))
        undoDeferTerminationSlow(deferAction);
}

ALWAYS_INLINE DeferTraps::DeferTraps(VM& vm)
    : m_traps(vm.traps())
    , m_isActive(!m_traps.hasTrapBit(VMTraps::DeferTrapHandling))
{
    if (m_isActive)
        m_traps.setTrapBit(VMTraps::DeferTrapHandling);
}

ALWAYS_INLINE DeferTraps::~DeferTraps()
{
    if (m_isActive)
        m_traps.clearTrapBit(VMTraps::DeferTrapHandling);
}

} // namespace JSC
