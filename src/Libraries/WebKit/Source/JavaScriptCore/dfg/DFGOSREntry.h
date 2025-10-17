/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 18, 2025.
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

#include "CodeLocation.h"
#include "DFGAbstractValue.h"
#include "DFGFlushFormat.h"
#include "MacroAssemblerCodeRef.h"
#include "Operands.h"
#include <wtf/BitVector.h>

namespace JSC {

class CallFrame;
class CodeBlock;

namespace DFG {

#if ENABLE(DFG_JIT)
struct OSREntryReshuffling {
    OSREntryReshuffling() { }
    
    OSREntryReshuffling(int fromOffset, int toOffset)
        : fromOffset(fromOffset)
        , toOffset(toOffset)
    {
    }
    
    int fromOffset;
    int toOffset;
};

struct OSREntryData {
    BytecodeIndex m_bytecodeIndex;
    CodeLocationLabel<OSREntryPtrTag> m_machineCode;
    FixedOperands<AbstractValue> m_expectedValues;
    // Use bitvectors here because they tend to only require one word.
    BitVector m_localsForcedDouble;
    BitVector m_localsForcedAnyInt;
    FixedVector<OSREntryReshuffling> m_reshufflings;
    BitVector m_machineStackUsed;
    
    void dumpInContext(PrintStream&, DumpContext*) const;
    void dump(PrintStream&) const;
};

inline BytecodeIndex getOSREntryDataBytecodeIndex(OSREntryData* osrEntryData)
{
    return osrEntryData->m_bytecodeIndex;
}

struct CatchEntrypointData {
    // We use this when doing OSR entry at catch. We prove the arguments
    // are of the expected type before entering at a catch block.
    CodePtr<ExceptionHandlerPtrTag> machineCode;
    FixedVector<FlushFormat> argumentFormats;
    BytecodeIndex bytecodeIndex;
};

// Returns a pointer to a data buffer that the OSR entry thunk will recognize and
// parse. If this returns null, it means 
void* prepareOSREntry(VM&, CallFrame*, CodeBlock*, BytecodeIndex);

// If null is returned, we can't OSR enter. If it's not null, it's the PC to jump to.
CodePtr<ExceptionHandlerPtrTag> prepareCatchOSREntry(VM&, CallFrame*, CodeBlock* baselineCodeBlock, CodeBlock* optimizedCodeBlock, BytecodeIndex);
#else
inline CodePtr<ExceptionHandlerPtrTag> prepareOSREntry(VM&, CallFrame*, CodeBlock*, BytecodeIndex) { return nullptr; }
#endif

} } // namespace JSC::DFG
