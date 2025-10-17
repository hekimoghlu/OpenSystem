/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 29, 2021.
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
#include "Yarr.h"
#include "YarrJIT.h"

namespace JSC {

class VM;
class ExecutablePool;
class RegExp;

namespace Yarr {

class MatchingContextHolder {
    WTF_FORBID_HEAP_ALLOCATION;
public:
    MatchingContextHolder(VM&, bool, RegExp*, MatchFrom);
    ~MatchingContextHolder();

    static constexpr ptrdiff_t offsetOfStackLimit() { return OBJECT_OFFSETOF(MatchingContextHolder, m_stackLimit); }
#if ENABLE(YARR_JIT_ALL_PARENS_EXPRESSIONS)
    static constexpr ptrdiff_t offsetOfPatternContextBuffer() { return OBJECT_OFFSETOF(MatchingContextHolder, m_patternContextBuffer); }
    static constexpr ptrdiff_t offsetOfPatternContextBufferSize() { return OBJECT_OFFSETOF(MatchingContextHolder, m_patternContextBufferSize); }
#endif

private:
    VM& m_vm;
    void* m_stackLimit;
#if ENABLE(YARR_JIT_ALL_PARENS_EXPRESSIONS)
    void* m_patternContextBuffer { nullptr };
    unsigned m_patternContextBufferSize { 0 };
#endif
    MatchFrom m_matchFrom;
};

inline MatchingContextHolder::MatchingContextHolder(VM& vm, bool usesPatternContextBuffer, RegExp* regExp, MatchFrom matchFrom)
    : m_vm(vm)
    , m_matchFrom(matchFrom)
{
    if (matchFrom == MatchFrom::VMThread) {
        m_stackLimit = vm.softStackLimit();
        vm.m_executingRegExp = regExp;
    } else {
        StackBounds stack = Thread::current().stack();
        m_stackLimit = stack.recursionLimit(Options::reservedZoneSize());
    }

#if ENABLE(YARR_JIT_ALL_PARENS_EXPRESSIONS)
    if (usesPatternContextBuffer) {
        m_patternContextBuffer = m_vm.acquireRegExpPatternContexBuffer();
        m_patternContextBufferSize = VM::patternContextBufferSize;
    }
#else
    UNUSED_PARAM(usesPatternContextBuffer);
#endif
}

inline MatchingContextHolder::~MatchingContextHolder()
{
#if ENABLE(YARR_JIT_ALL_PARENS_EXPRESSIONS)
    if (m_patternContextBuffer)
        m_vm.releaseRegExpPatternContexBuffer();
#endif
    if (m_matchFrom == MatchFrom::VMThread)
        m_vm.m_executingRegExp = nullptr;
}

} } // namespace JSC::Yarr
