/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 10, 2025.
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

#include "BaselineJITCode.h"
#include "BytecodeStructs.h"
#include "CodeBlock.h"
#include "DFGJITCode.h"
#include "UnlinkedMetadataTableInlines.h"

namespace JSC {

#define CODEBLOCK_MAGIC 0xc0deb10c

template<typename Functor>
void CodeBlock::forEachValueProfile(const Functor& func)
{
    for (auto& profile : argumentValueProfiles())
        func(profile, true);

    if (m_metadata) {
        auto wrapper = [&] (ValueProfile& profile) {
            func(profile, false);
        };
        m_metadata->forEachValueProfile(wrapper);
    }
}

template<typename Functor>
void CodeBlock::forEachArrayAllocationProfile(const Functor& func)
{
    if (m_metadata) {
#define VISIT(__op) \
    m_metadata->forEach<__op>([&] (auto& metadata) { func(metadata.m_arrayAllocationProfile); });

        FOR_EACH_OPCODE_WITH_ARRAY_ALLOCATION_PROFILE(VISIT)

#undef VISIT
    }
}

template<typename Functor>
void CodeBlock::forEachObjectAllocationProfile(const Functor& func)
{
    if (m_metadata) {
#define VISIT(__op) \
    m_metadata->forEach<__op>([&] (auto& metadata) { func(metadata.m_objectAllocationProfile); });

        FOR_EACH_OPCODE_WITH_OBJECT_ALLOCATION_PROFILE(VISIT)

#undef VISIT
    }
}

template<typename Functor>
void CodeBlock::forEachLLIntOrBaselineCallLinkInfo(const Functor& func)
{
    if (m_metadata) {
#define VISIT(__op) \
    m_metadata->forEach<__op>([&] (auto& metadata) { func(metadata.m_callLinkInfo); });

        FOR_EACH_OPCODE_WITH_CALL_LINK_INFO(VISIT)

#undef VISIT
    }
}

#if ENABLE(JIT)
ALWAYS_INLINE const JITCodeMap& CodeBlock::jitCodeMap()
{
    ASSERT(jitType() == JITType::BaselineJIT);
    return static_cast<BaselineJITCode*>(m_jitCode.get())->m_jitCodeMap;
}

ALWAYS_INLINE SimpleJumpTable& CodeBlock::baselineSwitchJumpTable(int tableIndex)
{
    ASSERT(jitType() == JITType::BaselineJIT);
    return static_cast<BaselineJITCode*>(m_jitCode.get())->m_switchJumpTables[tableIndex];
}

ALWAYS_INLINE StringJumpTable& CodeBlock::baselineStringSwitchJumpTable(int tableIndex)
{
    ASSERT(jitType() == JITType::BaselineJIT);
    return static_cast<BaselineJITCode*>(m_jitCode.get())->m_stringSwitchJumpTables[tableIndex];
}
#endif

#if ENABLE(DFG_JIT)
ALWAYS_INLINE SimpleJumpTable& CodeBlock::dfgSwitchJumpTable(int tableIndex)
{
    ASSERT(jitType() == JITType::DFGJIT);
    return static_cast<DFG::JITCode*>(m_jitCode.get())->m_switchJumpTables[tableIndex];
}

ALWAYS_INLINE StringJumpTable& CodeBlock::dfgStringSwitchJumpTable(int tableIndex)
{
    ASSERT(jitType() == JITType::DFGJIT);
    return static_cast<DFG::JITCode*>(m_jitCode.get())->m_stringSwitchJumpTables[tableIndex];
}
#endif

#if ASSERT_ENABLED
ALWAYS_INLINE bool CodeBlock::wasDestructed()
{
    return m_magic != CODEBLOCK_MAGIC;
}
#endif

} // namespace JSC
