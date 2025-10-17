/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 6, 2022.
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

#include "JSGlobalObject.h"
#include "SlotVisitor.h"
#include "WriteBarrier.h"

namespace JSC {

class InternalFunctionAllocationProfile {
public:
    static constexpr ptrdiff_t offsetOfStructureID() { return OBJECT_OFFSETOF(InternalFunctionAllocationProfile, m_structureID); }

    Structure* structure() { return m_structureID.get(); }
    Structure* createAllocationStructureFromBase(VM&, JSGlobalObject*, JSCell* owner, JSObject* prototype, Structure* base, InlineWatchpointSet&);

    void clear() { m_structureID.clear(); }
    template<typename Visitor> void visitAggregate(Visitor& visitor) { visitor.append(m_structureID); }

private:
    WriteBarrierStructureID m_structureID;
};

inline Structure* InternalFunctionAllocationProfile::createAllocationStructureFromBase(VM& vm, JSGlobalObject* baseGlobalObject, JSCell* owner, JSObject* prototype, Structure* baseStructure, InlineWatchpointSet& watchpointSet)
{
    ASSERT(!m_structureID || m_structureID.get()->classInfoForCells() != baseStructure->classInfoForCells() || m_structureID->globalObject() != baseStructure->globalObject());
    ASSERT(baseStructure->hasMonoProto());

    Structure* structure;
    // FIXME: Implement polymorphic prototypes for subclasses of builtin types:
    // https://bugs.webkit.org/show_bug.cgi?id=177318
    if (prototype == baseStructure->storedPrototype())
        structure = baseStructure;
    else
        structure = baseGlobalObject->structureCache().emptyStructureForPrototypeFromBaseStructure(baseGlobalObject, prototype, baseStructure);

    // Ensure that if another thread sees the structure, it will see it properly created.
    WTF::storeStoreFence();

    // It's possible to get here because some JSFunction got passed to two different InternalFunctions. e.g.
    // function Foo() { }
    // Reflect.construct(Promise, [], Foo);
    // Reflect.construct(Int8Array, [], Foo);
    if (UNLIKELY(m_structureID && m_structureID.value() != structure->id()))
        watchpointSet.fireAll(vm, "InternalFunctionAllocationProfile rotated to a new structure");

    m_structureID.set(vm, owner, structure);
    return structure;
}

} // namespace JSC
