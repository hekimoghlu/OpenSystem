/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 21, 2024.
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

#include "CCallHelpers.h"

namespace JSC {

class SnippetParams;

typedef CCallHelpers::JumpList SnippetCompilerFunction(CCallHelpers&, SnippetParams&);
typedef SharedTask<SnippetCompilerFunction> SnippetCompiler;

// Snippet is the way to inject an opaque code generator into DFG and FTL.
// While B3::Patchpoint is self-contained about its compilation information,
// Snippet depends on which DFG Node invokes. For example, CheckDOM will
// link returned failureCases to BadType OSRExit, but this information is offered
// from CheckDOM DFG Node, not from this snippet. This snippet mainly focuses
// on injecting a snippet generator that can tell register usage and can be used
// in both DFG and FTL.
class Snippet : public ThreadSafeRefCounted<Snippet> {
public:
    static Ref<Snippet> create()
    {
        return adoptRef(*new Snippet());
    }

    template<typename Functor>
    void setGenerator(const Functor& functor)
    {
        m_generator = createSharedTask<SnippetCompilerFunction>(functor);
    }

    RefPtr<SnippetCompiler> generator() const { return m_generator; }

    uint8_t numGPScratchRegisters { 0 };
    uint8_t numFPScratchRegisters { 0 };

protected:
    Snippet() = default;

private:
    RefPtr<SnippetCompiler> m_generator;
};

}

#endif
