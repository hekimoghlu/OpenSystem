/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 28, 2022.
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
#include "DFGSnippetParams.h"

#if ENABLE(DFG_JIT)

#include "DFGSlowPathGenerator.h"
#include "DFGSpeculativeJIT.h"
#include "JSCJSValueInlines.h"

namespace JSC { namespace DFG {

template<typename OperationType, typename ResultType, typename Arguments, size_t... ArgumentsIndex>
static void dispatch(SpeculativeJIT* jit, CCallHelpers::JumpList from, OperationType operation, ResultType result, Arguments arguments, std::index_sequence<ArgumentsIndex...>)
{
    jit->addSlowPathGenerator(slowPathCall(from, jit, operation, result, std::get<ArgumentsIndex>(arguments)...));
}

#define JSC_DEFINE_CALL_OPERATIONS(OperationType, ResultType, ...) \
    void SnippetParams::addSlowPathCallImpl(CCallHelpers::JumpList from, CCallHelpers&, OperationType operation, ResultType result, std::tuple<__VA_ARGS__> args) \
    { \
        dispatch(m_jit, from, operation, result, args, std::make_index_sequence<std::tuple_size<decltype(args)>::value>()); \
    } \

SNIPPET_SLOW_PATH_CALLS(JSC_DEFINE_CALL_OPERATIONS)
#undef JSC_DEFINE_CALL_OPERATIONS

} }

#endif
