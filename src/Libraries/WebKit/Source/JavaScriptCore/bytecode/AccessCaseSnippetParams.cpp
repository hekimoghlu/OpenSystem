/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 5, 2025.
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
#include "config.h"
#include "AccessCaseSnippetParams.h"

#include "InlineCacheCompiler.h"
#include "LinkBuffer.h"
#include "StructureStubInfo.h"

#if ENABLE(JIT)

namespace JSC {

template<typename JumpType, typename FunctionType, typename ResultType, typename... Arguments>
class SlowPathCallGeneratorWithArguments final : public AccessCaseSnippetParams::SlowPathCallGenerator {
public:
    SlowPathCallGeneratorWithArguments(JumpType from, CCallHelpers::Label to, FunctionType function, ResultType result, std::tuple<Arguments...> arguments)
        : m_from(from)
        , m_to(to)
        , m_function(function)
        , m_result(result)
        , m_arguments(arguments)
    {
    }

    template<size_t... ArgumentsIndex>
    CCallHelpers::JumpList generateImpl(InlineCacheCompiler& compiler, const RegisterSetBuilder& usedRegistersBySnippet, CCallHelpers& jit, std::index_sequence<ArgumentsIndex...>)
    {
        CCallHelpers::JumpList exceptions;
        // We spill (1) the used registers by IC and (2) the used registers by Snippet.
        InlineCacheCompiler::SpillState spillState = compiler.preserveLiveRegistersToStackForCall(usedRegistersBySnippet.buildAndValidate());

        jit.makeSpaceOnStackForCCall();

        jit.setupArguments<FunctionType>(std::get<ArgumentsIndex>(m_arguments)...);
        jit.prepareCallOperation(compiler.vm());
        jit.callOperation<OperationPtrTag>(m_function);
        jit.setupResults(m_result);
        jit.reclaimSpaceOnStackForCCall();

        CCallHelpers::Jump noException = jit.emitExceptionCheck(compiler.vm(), CCallHelpers::InvertedExceptionCheck);

        compiler.restoreLiveRegistersFromStackForCallWithThrownException(spillState);
        exceptions.append(jit.jump());

        noException.link(&jit);
        RegisterSet dontRestore;
        dontRestore.add(m_result, IgnoreVectors);
        compiler.restoreLiveRegistersFromStackForCall(spillState, dontRestore);

        return exceptions;
    }

    CCallHelpers::JumpList generate(InlineCacheCompiler& compiler, const RegisterSetBuilder& usedRegistersBySnippet, CCallHelpers& jit) final
    {
        m_from.link(&jit);
        CCallHelpers::JumpList exceptions = generateImpl(compiler, usedRegistersBySnippet, jit, std::make_index_sequence<std::tuple_size<std::tuple<Arguments...>>::value>());
        jit.jump().linkTo(m_to, &jit);
        return exceptions;
    }

private:
    JumpType m_from;
    CCallHelpers::Label m_to;
    FunctionType m_function;
    ResultType m_result;
    std::tuple<Arguments...> m_arguments;
};

#define JSC_DEFINE_CALL_OPERATIONS(OperationType, ResultType, ...) \
    void AccessCaseSnippetParams::addSlowPathCallImpl(CCallHelpers::JumpList from, CCallHelpers& jit, OperationType operation, ResultType result, std::tuple<__VA_ARGS__> args) \
    { \
        CCallHelpers::Label to = jit.label(); \
        m_generators.append(makeUnique<SlowPathCallGeneratorWithArguments<CCallHelpers::JumpList, OperationType, ResultType, __VA_ARGS__>>(from, to, operation, result, args)); \
    } \

SNIPPET_SLOW_PATH_CALLS(JSC_DEFINE_CALL_OPERATIONS)
#undef JSC_DEFINE_CALL_OPERATIONS

CCallHelpers::JumpList AccessCaseSnippetParams::emitSlowPathCalls(InlineCacheCompiler& compiler, const RegisterSetBuilder& usedRegistersBySnippet, CCallHelpers& jit)
{
    CCallHelpers::JumpList exceptions;
    for (auto& generator : m_generators)
        exceptions.append(generator->generate(compiler, usedRegistersBySnippet, jit));
    return exceptions;
}

}

#endif
