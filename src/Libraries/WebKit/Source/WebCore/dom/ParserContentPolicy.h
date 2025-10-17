/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 20, 2022.
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

#include <wtf/OptionSet.h>

namespace WebCore {

enum class ParserContentPolicy : uint8_t {
    AllowScriptingContent = 1 << 0,
    DoNotMarkAlreadyStarted = 1 << 1,
    AllowDeclarativeShadowRoots = 1 << 2,
    AlwaysParseAsHTML = 1 << 3
};

constexpr OptionSet<ParserContentPolicy> DefaultParserContentPolicy = { ParserContentPolicy::AllowScriptingContent, ParserContentPolicy::AllowDeclarativeShadowRoots };

static inline bool scriptingContentIsAllowed(OptionSet<ParserContentPolicy> parserContentPolicy) 
{
    return parserContentPolicy.contains(ParserContentPolicy::AllowScriptingContent);
}

static inline OptionSet<ParserContentPolicy> disallowScriptingContent(OptionSet<ParserContentPolicy> parserContentPolicy)
{
    parserContentPolicy.remove(ParserContentPolicy::AllowScriptingContent);
    return parserContentPolicy;
}

} // namespace WebCore
