/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 24, 2023.
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
#include "StructureChain.h"

#include "JSCInlines.h"

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace JSC {
    
const ClassInfo StructureChain::s_info = { "StructureChain"_s, nullptr, nullptr, nullptr, CREATE_METHOD_TABLE(StructureChain) };

StructureChain::StructureChain(VM& vm, Structure* structure, StructureID* vector)
    : Base(vm, structure)
    , m_vector(vector, WriteBarrierEarlyInit)
{
}

StructureChain* StructureChain::create(VM& vm, JSObject* head)
{
    // FIXME: Make StructureChain::create fail for large chain. Caching large chain is not so profitable.
    // By making the size <= UINT16_MAX, we can store length in a high bits of auxiliary pointer.
    // https://bugs.webkit.org/show_bug.cgi?id=200290
    size_t size = 0;
    for (JSObject* current = head; current; current = current->structure()->storedPrototypeObject(current))
        ++size;
    ++size; // Sentinel nullptr.
    size_t bytes = Checked<size_t>(size) * sizeof(StructureID);
    void* vector = vm.auxiliarySpace().allocate(vm, bytes, nullptr, AllocationFailureMode::Assert);
    static_assert(!StructureID().bits(), "Make sure the value we're going to memcpy below matches the default StructureID");
    memset(vector, 0, bytes);
    StructureChain* chain = new (NotNull, allocateCell<StructureChain>(vm)) StructureChain(vm, vm.structureChainStructure.get(), static_cast<StructureID*>(vector));
    chain->finishCreation(vm, head);
    return chain;
}

void StructureChain::finishCreation(VM& vm, JSObject* head)
{
    Base::finishCreation(vm);
    size_t i = 0;
    for (JSObject* current = head; current; current = current->structure()->storedPrototypeObject(current)) {
        Structure* structure = current->structure();
        m_vector.get()[i++] = structure->id();
        vm.writeBarrier(this);
    }
}

template<typename Visitor>
void StructureChain::visitChildrenImpl(JSCell* cell, Visitor& visitor)
{
    StructureChain* thisObject = jsCast<StructureChain*>(cell);
    ASSERT_GC_OBJECT_INHERITS(thisObject, info());
    Base::visitChildren(thisObject, visitor);
    visitor.markAuxiliary(thisObject->m_vector.get());
    for (auto* current = thisObject->m_vector.get(); *current; ++current) {
        StructureID structureID = *current;
        Structure* structure = structureID.decode();
        visitor.appendUnbarriered(structure);
    }
}

DEFINE_VISIT_CHILDREN(StructureChain);

} // namespace JSC

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
