/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 1, 2021.
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

#if ENABLE(B3_JIT)

#include "B3FrequencyClass.h"
#include <wtf/PrintStream.h>

namespace JSC { namespace B3 {

// A frequented block is a tuple of BasicBlock* and FrequencyClass. It's usually used as a
// successor edge.

template<typename BasicBlock>
class GenericFrequentedBlock {
public:
    GenericFrequentedBlock(
        BasicBlock* block = nullptr, FrequencyClass frequency = FrequencyClass::Normal)
        : m_block(block)
        , m_frequency(frequency)
    {
    }

    friend bool operator==(const GenericFrequentedBlock&, const GenericFrequentedBlock&) = default;

    explicit operator bool() const
    {
        return *this != GenericFrequentedBlock();
    }

    BasicBlock* block() const { return m_block; }
    BasicBlock*& block() { return m_block; }
    FrequencyClass frequency() const { return m_frequency; }
    FrequencyClass& frequency() { return m_frequency; }

    bool isRare() const { return frequency() == FrequencyClass::Rare; }

    void dump(PrintStream& out) const
    {
        if (frequency() != FrequencyClass::Normal)
            out.print(frequency(), ":");
        out.print(pointerDump(m_block));
    }

private:
    BasicBlock* m_block;
    FrequencyClass m_frequency;
};

} } // namespace JSC::B3

#endif // ENABLE(B3_JIT)
