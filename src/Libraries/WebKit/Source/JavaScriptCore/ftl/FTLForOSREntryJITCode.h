/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 28, 2023.
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

#if ENABLE(FTL_JIT)

#include "FTLJITCode.h"
#include <wtf/FixedVector.h>

namespace JSC { namespace FTL {

// OSR entry into the FTL has a number of quirks:
//
// - OSR entry only happens through special OSR entry compilations. They have their
//   own CodeBlock and their own JITCode.
//
// - We only OSR enter in loop headers that have a null inline call frame.
//
// - Each OSR entry compilation allows entry through only one bytecode index.

class ForOSREntryJITCode final : public FTL::JITCode {
public:
    ForOSREntryJITCode();
    ~ForOSREntryJITCode() final;
    
    void initializeEntryBuffer(VM&, unsigned numCalleeLocals);
    ScratchBuffer* entryBuffer() const { return m_entryBuffer; }
    
    void setBytecodeIndex(BytecodeIndex value) { m_bytecodeIndex = value; }
    BytecodeIndex bytecodeIndex() const { return m_bytecodeIndex; }
    
    void countEntryFailure() { m_entryFailureCount++; }
    unsigned entryFailureCount() const { return m_entryFailureCount; }
    
    ForOSREntryJITCode* ftlForOSREntry() final;

    void setArgumentFlushFormats(FixedVector<DFG::FlushFormat>&& argumentFlushFormats)
    {
        m_argumentFlushFormats = WTFMove(argumentFlushFormats);
    }
    const FixedVector<DFG::FlushFormat>& argumentFlushFormats() { return m_argumentFlushFormats; }

private:
    FixedVector<DFG::FlushFormat> m_argumentFlushFormats;
    ScratchBuffer* m_entryBuffer; // Only for OSR entry code blocks.
    BytecodeIndex m_bytecodeIndex;
    unsigned m_entryFailureCount;
};

} } // namespace JSC::FTL

#endif // ENABLE(FTL_JIT)
