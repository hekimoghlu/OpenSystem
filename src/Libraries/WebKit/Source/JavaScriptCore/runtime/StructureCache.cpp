/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 21, 2022.
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
#include "StructureCache.h"

#include "JSCInlines.h"
#include <wtf/Locker.h>

namespace JSC {

void StructureCache::clear()
{
    Locker locker { m_lock };
    m_structures.clear();
}

inline Structure* StructureCache::createEmptyStructure(JSGlobalObject* globalObject, JSObject* prototype, const TypeInfo& typeInfo, const ClassInfo* classInfo, IndexingType indexingType, unsigned inlineCapacity, bool makePolyProtoStructure, FunctionExecutable* executable)
{
    RELEASE_ASSERT(!!prototype); // We use nullptr inside the UncheckedKeyHashMap for prototype to mean poly proto, so user's of this API must provide non-null prototypes.

    // We don't need to lock here because only the main thread can get here, and only the main thread can mutate the cache
    ASSERT(!isCompilationThread() && !Thread::mayBeGCThread());
    VM& vm = globalObject->vm();
    PrototypeKey key { makePolyProtoStructure ? nullptr : prototype, executable, inlineCapacity, classInfo };
    if (Structure* structure = m_structures.get(key)) {
        if (makePolyProtoStructure) {
            prototype->didBecomePrototype(vm);
            RELEASE_ASSERT(structure->hasPolyProto());
        } else
            RELEASE_ASSERT(structure->hasMonoProto());
        ASSERT(prototype->mayBePrototype());
        return structure;
    }

    prototype->didBecomePrototype(vm);

    Structure* structure;
    if (makePolyProtoStructure) {
        structure = Structure::create(
            Structure::PolyProto, vm, globalObject, prototype, typeInfo, classInfo, indexingType, inlineCapacity);
    } else {
        structure = Structure::create(
            vm, globalObject, prototype, typeInfo, classInfo, indexingType, inlineCapacity);
    }
    Locker locker { m_lock };
    m_structures.set(key, structure);
    return structure;
}

Structure* StructureCache::emptyObjectStructureConcurrently(JSObject* prototype, unsigned inlineCapacity)
{
    RELEASE_ASSERT(!!prototype); // We use nullptr inside the UncheckedKeyHashMap for prototype to mean poly proto, so user's of this API must provide non-null prototypes.
    
    PrototypeKey key { prototype, nullptr, inlineCapacity, JSFinalObject::info() };
    Locker locker { m_lock };
    if (Structure* structure = m_structures.get(key)) {
        ASSERT(prototype->mayBePrototype());
        return structure;
    }
    return nullptr;
}

Structure* StructureCache::emptyStructureForPrototypeFromBaseStructure(JSGlobalObject* globalObject, JSObject* prototype, Structure* baseStructure)
{
    // We currently do not have inline capacity static analysis for subclasses and all internal function constructors have a default inline capacity of 0.
    IndexingType indexingType = baseStructure->indexingType();
    if (prototype->anyObjectInChainMayInterceptIndexedAccesses() && hasIndexedProperties(indexingType))
        indexingType = (indexingType & ~IndexingShapeMask) | SlowPutArrayStorageShape;

    return createEmptyStructure(globalObject, prototype, baseStructure->typeInfo(), baseStructure->classInfoForCells(), indexingType, 0, false, nullptr);
}

Structure* StructureCache::emptyObjectStructureForPrototype(JSGlobalObject* globalObject, JSObject* prototype, unsigned inlineCapacity, bool makePolyProtoStructure, FunctionExecutable* executable)
{
    return createEmptyStructure(globalObject, prototype, JSFinalObject::typeInfo(), JSFinalObject::info(), JSFinalObject::defaultIndexingType, inlineCapacity, makePolyProtoStructure, executable);
}

} // namespace JSC
