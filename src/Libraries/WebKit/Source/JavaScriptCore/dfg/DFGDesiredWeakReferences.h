/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 26, 2024.
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

#if ENABLE(DFG_JIT)

#include "StructureID.h"
#include "WriteBarrier.h"
#include <wtf/FixedVector.h>
#include <wtf/HashSet.h>

namespace JSC {

class CodeBlock;
class JSCell;
class JSValue;
class VM;

namespace DFG {

class CommonData;

class DesiredWeakReferences {
public:
    DesiredWeakReferences();
    DesiredWeakReferences(CodeBlock*);
    ~DesiredWeakReferences();

    void addLazily(JSCell*);
    void addLazily(JSValue);
    bool contains(JSCell*);

    void reallyAdd(VM&, CommonData*);

    void finalize();

    template<typename Visitor> void visitChildren(Visitor&);

private:
    CodeBlock* m_codeBlock;
    UncheckedKeyHashSet<JSCell*> m_cells;
    UncheckedKeyHashSet<StructureID> m_structures;
    FixedVector<WriteBarrier<JSCell>> m_finalizedCells;
    FixedVector<StructureID> m_finalizedStructures;
};

} } // namespace JSC::DFG

#endif // ENABLE(DFG_JIT)
