/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 21, 2023.
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

#include <wtf/Platform.h>

#if ENABLE(JIT)

#include "SnippetParams.h"
#include <wtf/TZoneMalloc.h>

namespace JSC {

class InlineCacheCompiler;

class AccessCaseSnippetParams final : public SnippetParams {
public:
    friend class InlineCacheCompiler;
    AccessCaseSnippetParams(VM& vm, Vector<Value>&& regs, Vector<GPRReg>&& gpScratch, Vector<FPRReg>&& fpScratch)
        : SnippetParams(vm, WTFMove(regs), WTFMove(gpScratch), WTFMove(fpScratch))
    {
    }

    class SlowPathCallGenerator {
        WTF_MAKE_FAST_ALLOCATED;
    public:
        virtual ~SlowPathCallGenerator() { }
        virtual CCallHelpers::JumpList generate(InlineCacheCompiler&, const RegisterSetBuilder& usedRegistersBySnippet, CCallHelpers&) = 0;
    };

    CCallHelpers::JumpList emitSlowPathCalls(InlineCacheCompiler&, const RegisterSetBuilder& usedRegistersBySnippet, CCallHelpers&);

private:
#define JSC_DEFINE_CALL_OPERATIONS(OperationType, ResultType, ...) void addSlowPathCallImpl(CCallHelpers::JumpList, CCallHelpers&, OperationType, ResultType, std::tuple<__VA_ARGS__> args) final;
    SNIPPET_SLOW_PATH_CALLS(JSC_DEFINE_CALL_OPERATIONS)
#undef JSC_DEFINE_CALL_OPERATIONS
    Vector<std::unique_ptr<SlowPathCallGenerator>> m_generators;
};

}

#endif
