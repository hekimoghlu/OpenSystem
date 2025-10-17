/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 11, 2025.
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
#include "LazyClassStructure.h"

#include "JSCInlines.h"
#include "LazyPropertyInlines.h"

namespace JSC {

LazyClassStructure::Initializer::Initializer(VM& vm, JSGlobalObject* global, LazyClassStructure& classStructure, const StructureInitializer& structureInit)
    : vm(vm)
    , global(global)
    , classStructure(classStructure)
    , structureInit(structureInit)
{
}

void LazyClassStructure::Initializer::setPrototype(JSObject* prototype)
{
    RELEASE_ASSERT(!this->prototype);
    RELEASE_ASSERT(!structure);
    RELEASE_ASSERT(!constructor);
    
    this->prototype = prototype;
}

void LazyClassStructure::Initializer::setStructure(Structure* structure)
{
    RELEASE_ASSERT(!this->structure);
    RELEASE_ASSERT(!constructor);

    this->structure = structure;
    structureInit.set(structure);
    
    if (!prototype)
        prototype = structure->storedPrototypeObject();
}

void LazyClassStructure::Initializer::setConstructor(JSObject* constructor)
{
    RELEASE_ASSERT(structure);
    RELEASE_ASSERT(prototype);
    RELEASE_ASSERT(!this->constructor);
    
    this->constructor = constructor;

    prototype->putDirectWithoutTransition(vm, vm.propertyNames->constructor, constructor, static_cast<unsigned>(PropertyAttribute::DontEnum));
    classStructure.m_constructor.set(vm, global, constructor);
}

template<typename Visitor>
void LazyClassStructure::visit(Visitor& visitor)
{
    m_structure.visit(visitor);
    visitor.append(m_constructor);
}

template void LazyClassStructure::visit(AbstractSlotVisitor&);
template void LazyClassStructure::visit(SlotVisitor&);

void LazyClassStructure::dump(PrintStream& out) const
{
    out.print("<structure = ", m_structure, ", constructor = ", RawPointer(m_constructor.get()), ">");
}

} // namespace JSC

