/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 16, 2024.
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

#include <wtf/Vector.h>

#if ENABLE(DFG_JIT)

namespace JSC { 

class CodeBlock;
class ScriptExecutable;
class Structure;
class VM;

namespace DFG {

class CommonData;
class DesiredTransitions;

class DesiredTransition {
public:
    friend class DesiredTransitions;
    DesiredTransition(CodeBlock* codeOriginOwner, Structure*, Structure*);

    template<typename Visitor> void visitChildren(Visitor&);

private:
    CodeBlock* m_codeOriginOwner;
    Structure* m_oldStructure;
    Structure* m_newStructure;
};

class DesiredTransitions {
public:
    DesiredTransitions() = default;
    DesiredTransitions(CodeBlock*);
    ~DesiredTransitions();

    void addLazily(CodeBlock* codeOriginOwner, Structure*, Structure*);
    void reallyAdd(VM&, CommonData*);
    template<typename Visitor> void visitChildren(Visitor&);

private:
    CodeBlock* m_codeBlock { nullptr };
    Vector<DesiredTransition> m_transitions;
};

} } // namespace JSC::DFG

#endif // ENABLE(DFG_JIT)
