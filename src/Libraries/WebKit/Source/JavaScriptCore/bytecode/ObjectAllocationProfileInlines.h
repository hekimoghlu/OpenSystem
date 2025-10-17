/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 4, 2022.
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

#include "ObjectAllocationProfile.h"

#include "JSFunctionInlines.h"

namespace JSC {

template<typename Derived>
ALWAYS_INLINE void ObjectAllocationProfileBase<Derived>::initializeProfile(VM& vm, JSGlobalObject* globalObject, JSCell* owner, JSObject* prototype, unsigned inferredInlineCapacity, JSFunction* constructor, FunctionRareData* functionRareData)
{
    ASSERT(!m_allocator);
    ASSERT(!m_structure);

    // FIXME: Teach create_this's fast path how to allocate poly
    // proto objects: https://bugs.webkit.org/show_bug.cgi?id=177517

    bool isPolyProto = false;
    FunctionExecutable* executable = nullptr;
    if (constructor) {
        // FIXME: A JSFunction should watch the poly proto watchpoint if it is not invalidated.
        // That way it can clear this object allocation profile to ensure it stops allocating
        // mono proto |this| values when it knows that it should be allocating poly proto
        // |this| values:
        // https://bugs.webkit.org/show_bug.cgi?id=177792

        executable = constructor->jsExecutable();

        if (Structure* structure = executable->cachedPolyProtoStructure()) {
            RELEASE_ASSERT(structure->typeInfo().type() == FinalObjectType);
            m_allocator = Allocator();
            m_structure.set(vm, owner, structure);
            static_cast<Derived*>(this)->setPrototype(vm, owner, prototype);
            return;
        }

        isPolyProto = false;
        if (Options::forcePolyProto())
            isPolyProto = true;
        else
            isPolyProto = executable->ensurePolyProtoWatchpoint().hasBeenInvalidated() && executable->singleton().hasBeenInvalidated();
    }

    unsigned inlineCapacity = 0;
    if (inferredInlineCapacity < JSFinalObject::defaultInlineCapacity) {
        // Try to shrink the object based on static analysis.
        inferredInlineCapacity += possibleDefaultPropertyCount(vm, prototype);

        if (!inferredInlineCapacity) {
            // Empty objects are rare, so most likely the static analyzer just didn't
            // see the real initializer function. This can happen with helper functions.
            inferredInlineCapacity = JSFinalObject::defaultInlineCapacity;
        } else if (inferredInlineCapacity > JSFinalObject::defaultInlineCapacity) {
            // Default properties are weak guesses, so don't allow them to turn a small
            // object into a large object.
            inferredInlineCapacity = JSFinalObject::defaultInlineCapacity;
        }

        inlineCapacity = inferredInlineCapacity;
        ASSERT(inlineCapacity < JSFinalObject::maxInlineCapacity);
    } else {
        // Normal or large object.
        inlineCapacity = inferredInlineCapacity;
        if (inlineCapacity > JSFinalObject::maxInlineCapacity)
            inlineCapacity = JSFinalObject::maxInlineCapacity;
    }

    if (isPolyProto) {
        ++inlineCapacity;
        inlineCapacity = std::min(inlineCapacity, JSFinalObject::maxInlineCapacity);
    }

    ASSERT(inlineCapacity > 0);
    ASSERT(inlineCapacity <= JSFinalObject::maxInlineCapacity);

    size_t allocationSize = JSFinalObject::allocationSize(inlineCapacity);
    Allocator allocator = subspaceFor<JSFinalObject>(vm)->allocatorFor(allocationSize, AllocatorForMode::EnsureAllocator);

    // Take advantage of extra inline capacity available in the size class.
    if (allocator) {
        size_t slop = (allocator.cellSize() - allocationSize) / sizeof(WriteBarrier<Unknown>);
        inlineCapacity += slop;
        if (inlineCapacity > JSFinalObject::maxInlineCapacity)
            inlineCapacity = JSFinalObject::maxInlineCapacity;
    }

    Structure* structure = globalObject->structureCache().emptyObjectStructureForPrototype(globalObject, prototype, inlineCapacity, isPolyProto, executable);

    if (isPolyProto) {
        ASSERT(structure->hasPolyProto());
        m_allocator = Allocator();
        executable->setCachedPolyProtoStructure(vm, structure);
    } else {
        if (executable) {
            ASSERT(constructor);
            ASSERT(functionRareData);
            InlineWatchpointSet& polyProtoWatchpointSet = executable->ensurePolyProtoWatchpoint();
            structure->ensureRareData(vm)->setSharedPolyProtoWatchpoint(executable->sharedPolyProtoWatchpoint());
            if (polyProtoWatchpointSet.isStillValid() && !functionRareData->hasAllocationProfileClearingWatchpoint()) {
                // If we happen to go poly proto in the future, we want to clear this particular
                // object allocation profile so we can transition to allocating poly proto objects.
                Watchpoint* watchpoint = functionRareData->createAllocationProfileClearingWatchpoint();
                polyProtoWatchpointSet.add(watchpoint);
            }
        }

        m_allocator = allocator;
    }

    // Ensure that if another thread sees the structure and prototype, it will see it properly created.
    WTF::storeStoreFence();

    // The watchpoint should have been fired already but it's prudent to be safe here.
    if (UNLIKELY(functionRareData && m_structure && m_structure.get() != structure)) {
        ASSERT(functionRareData->allocationProfileWatchpointSet().hasBeenInvalidated());
        functionRareData->allocationProfileWatchpointSet().fireAll(vm, "Clearing to be safe because structure has changed");
    }

    m_structure.set(vm, owner, structure);
    static_cast<Derived*>(this)->setPrototype(vm, owner, prototype);
}

template<typename Derived>
ALWAYS_INLINE unsigned ObjectAllocationProfileBase<Derived>::possibleDefaultPropertyCount(VM& vm, JSObject* prototype)
{
    if (prototype == prototype->globalObject()->objectPrototype())
        return 0;

    size_t count = 0;
    PropertyNameArray propertyNameArray(vm, PropertyNameMode::StringsAndSymbols, PrivateSymbolMode::Include);
    prototype->structure()->getPropertyNamesFromStructure(vm, propertyNameArray, DontEnumPropertiesMode::Include);
    PropertyNameArrayData::PropertyNameVector& propertyNameVector = propertyNameArray.data()->propertyNameVector();
    for (size_t i = 0; i < propertyNameVector.size(); ++i) {
        JSValue value = prototype->getDirect(vm, propertyNameVector[i]);

        // Functions are common, and are usually class-level objects that are not overridden.
        if (jsDynamicCast<JSFunction*>(value))
            continue;

        ++count;

    }
    return count;
}

} // namespace JSC
