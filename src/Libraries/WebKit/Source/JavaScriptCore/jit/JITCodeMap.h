/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 25, 2022.
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

#if ENABLE(JIT)

#include "BytecodeIndex.h"
#include "CodeLocation.h"
#include <wtf/MallocPtr.h>
#include <wtf/StdLibExtras.h>
#include <wtf/Vector.h>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace JSC {

DECLARE_ALLOCATOR_WITH_HEAP_IDENTIFIER(JITCodeMap);

class JITCodeMap {
public:
    static_assert(std::is_trivially_destructible_v<BytecodeIndex>);
    static_assert(std::is_trivially_destructible_v<CodeLocationLabel<JSEntryPtrTag>>);
    static_assert(alignof(CodeLocationLabel<JSEntryPtrTag>) >= alignof(BytecodeIndex), "Putting CodeLocationLabel vector first since we can avoid alignment consideration of BytecodeIndex vector");
    JITCodeMap() = default;
    JITCodeMap(Vector<BytecodeIndex>&& indexes, Vector<CodeLocationLabel<JSEntryPtrTag>>&& codeLocations)
        : m_size(indexes.size())
    {
        ASSERT(indexes.size() == codeLocations.size());
        m_pointer = MallocPtr<uint8_t, JITCodeMapMalloc>::malloc(sizeof(CodeLocationLabel<JSEntryPtrTag>) * m_size + sizeof(BytecodeIndex) * m_size);
        std::copy(codeLocations.begin(), codeLocations.end(), this->codeLocations());
        std::copy(indexes.begin(), indexes.end(), this->indexes());
    }

    CodeLocationLabel<JSEntryPtrTag> find(BytecodeIndex bytecodeIndex) const
    {
        auto* index = binarySearch<BytecodeIndex, BytecodeIndex>(indexes(), m_size, bytecodeIndex, [] (BytecodeIndex* index) { return *index; });
        if (!index)
            return CodeLocationLabel<JSEntryPtrTag>();
        return codeLocations()[index - indexes()];
    }

    explicit operator bool() const { return m_size; }

private:
    CodeLocationLabel<JSEntryPtrTag>* codeLocations() const
    {
        return std::bit_cast<CodeLocationLabel<JSEntryPtrTag>*>(m_pointer.get());
    }

    BytecodeIndex* indexes() const
    {
        return std::bit_cast<BytecodeIndex*>(m_pointer.get() + sizeof(CodeLocationLabel<JSEntryPtrTag>) * m_size);
    }

    MallocPtr<uint8_t, JITCodeMapMalloc> m_pointer;
    unsigned m_size { 0 };
};

class JITCodeMapBuilder {
    WTF_MAKE_NONCOPYABLE(JITCodeMapBuilder);
public:
    JITCodeMapBuilder() = default;
    void append(BytecodeIndex bytecodeIndex, CodeLocationLabel<JSEntryPtrTag> codeLocation)
    {
        m_indexes.append(bytecodeIndex);
        m_codeLocations.append(codeLocation);
    }

    JITCodeMap finalize()
    {
        return JITCodeMap(WTFMove(m_indexes), WTFMove(m_codeLocations));
    }

private:
    Vector<BytecodeIndex> m_indexes;
    Vector<CodeLocationLabel<JSEntryPtrTag>> m_codeLocations;
};

} // namespace JSC

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

#endif // ENABLE(JIT)
