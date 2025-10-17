/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 27, 2025.
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

#include "JSCPtrTag.h"

namespace JSC {

namespace LLInt {

// These are used just to denote where LLInt code begins and where it ends.
extern "C" {
    void llintPCRangeStart();
    void llintPCRangeEnd();
#if ENABLE(WEBASSEMBLY)
    void wasmLLIntPCRangeStart();
    void wasmLLIntPCRangeEnd();
#endif
}

ALWAYS_INLINE bool isLLIntPC(void* pc)
{
    uintptr_t pcAsInt = std::bit_cast<uintptr_t>(pc);
    uintptr_t llintStart = untagCodePtr<uintptr_t, CFunctionPtrTag>(llintPCRangeStart);
    uintptr_t llintEnd = untagCodePtr<uintptr_t, CFunctionPtrTag>(llintPCRangeEnd);
    RELEASE_ASSERT(llintStart < llintEnd);
    return llintStart <= pcAsInt && pcAsInt <= llintEnd;
}

#if ENABLE(WEBASSEMBLY)
ALWAYS_INLINE bool isWasmLLIntPC(void* pc)
{
    uintptr_t pcAsInt = std::bit_cast<uintptr_t>(pc);
    uintptr_t start = untagCodePtr<uintptr_t, CFunctionPtrTag>(wasmLLIntPCRangeStart);
    uintptr_t end = untagCodePtr<uintptr_t, CFunctionPtrTag>(wasmLLIntPCRangeEnd);
    RELEASE_ASSERT(start < end);
    return start <= pcAsInt && pcAsInt <= end;
}
#endif

#if !ENABLE(C_LOOP)
static constexpr GPRReg LLIntPC = GPRInfo::regT4;
#endif

} } // namespace JSC::LLInt
