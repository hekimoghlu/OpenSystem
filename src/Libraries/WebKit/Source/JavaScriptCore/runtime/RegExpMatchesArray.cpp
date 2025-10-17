/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 2, 2021.
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
#include "RegExpMatchesArray.h"

namespace JSC {

JSArray* createEmptyRegExpMatchesArray(JSGlobalObject* globalObject, JSString* input, RegExp* regExp)
{
    VM& vm = globalObject->vm();
    JSArray* array;

    // FIXME: This should handle array allocation errors gracefully.
    // https://bugs.webkit.org/show_bug.cgi?id=155144
    
    GCDeferralContext deferralContext(vm);
    ObjectInitializationScope scope(vm);

    if (UNLIKELY(globalObject->isHavingABadTime())) {
        array = JSArray::tryCreateUninitializedRestricted(scope, &deferralContext,
            regExp->hasIndices() ? globalObject->regExpMatchesArrayWithIndicesStructure() : globalObject->regExpMatchesArrayStructure(), regExp->numSubpatterns() + 1);
        // FIXME: we should probably throw an out of memory error here, but
        // when making this change we should check that all clients of this
        // function will correctly handle an exception being thrown from here.
        // https://bugs.webkit.org/show_bug.cgi?id=169786
        RELEASE_ASSERT(array);

        array->initializeIndexWithoutBarrier(scope, 0, jsEmptyString(vm));
        
        if (unsigned numSubpatterns = regExp->numSubpatterns()) {
            for (unsigned i = 1; i <= numSubpatterns; ++i)
                array->initializeIndexWithoutBarrier(scope, i, jsUndefined());
        }
    } else {
        array = tryCreateUninitializedRegExpMatchesArray(scope, &deferralContext,
            regExp->hasIndices() ? globalObject->regExpMatchesArrayWithIndicesStructure() : globalObject->regExpMatchesArrayStructure(), regExp->numSubpatterns() + 1);
        RELEASE_ASSERT(array);
        
        array->initializeIndexWithoutBarrier(scope, 0, jsEmptyString(vm), ArrayWithContiguous);
        
        if (unsigned numSubpatterns = regExp->numSubpatterns()) {
            for (unsigned i = 1; i <= numSubpatterns; ++i)
                array->initializeIndexWithoutBarrier(scope, i, jsUndefined(), ArrayWithContiguous);
        }
    }

    array->putDirectWithoutBarrier(RegExpMatchesArrayIndexPropertyOffset, jsNumber(-1));
    array->putDirectWithoutBarrier(RegExpMatchesArrayInputPropertyOffset, input);
    array->putDirectWithoutBarrier(RegExpMatchesArrayGroupsPropertyOffset, jsUndefined());
    if (regExp->hasIndices())
        array->putDirectWithoutBarrier(RegExpMatchesArrayIndicesPropertyOffset, jsUndefined());
    return array;
}

static Structure* createStructureImpl(VM& vm, JSGlobalObject* globalObject, IndexingType indexingType)
{
    Structure* structure = globalObject->arrayStructureForIndexingTypeDuringAllocation(indexingType);
    PropertyOffset offset;
    structure = Structure::addPropertyTransition(vm, structure, vm.propertyNames->index, 0, offset);
    ASSERT(offset == RegExpMatchesArrayIndexPropertyOffset);
    structure = Structure::addPropertyTransition(vm, structure, vm.propertyNames->input, 0, offset);
    ASSERT(offset == RegExpMatchesArrayInputPropertyOffset);
    structure = Structure::addPropertyTransition(vm, structure, vm.propertyNames->groups, 0, offset);
    ASSERT(offset == RegExpMatchesArrayGroupsPropertyOffset);
    return structure;
}

static Structure* createStructureWithIndicesImpl(VM& vm, JSGlobalObject* globalObject, IndexingType indexingType)
{
    Structure* structure = globalObject->arrayStructureForIndexingTypeDuringAllocation(indexingType);
    PropertyOffset offset;
    structure = Structure::addPropertyTransition(vm, structure, vm.propertyNames->index, 0, offset);
    ASSERT(offset == RegExpMatchesArrayIndexPropertyOffset);
    structure = Structure::addPropertyTransition(vm, structure, vm.propertyNames->input, 0, offset);
    ASSERT(offset == RegExpMatchesArrayInputPropertyOffset);
    structure = Structure::addPropertyTransition(vm, structure, vm.propertyNames->groups, 0, offset);
    ASSERT(offset == RegExpMatchesArrayGroupsPropertyOffset);
    structure = Structure::addPropertyTransition(vm, structure, vm.propertyNames->indices, 0, offset);
    ASSERT(offset == RegExpMatchesArrayIndicesPropertyOffset);
    return structure;
}

static Structure* createIndicesStructureImpl(VM& vm, JSGlobalObject* globalObject, IndexingType indexingType)
{
    Structure* structure = globalObject->arrayStructureForIndexingTypeDuringAllocation(indexingType);
    PropertyOffset offset;
    structure = Structure::addPropertyTransition(vm, structure, vm.propertyNames->groups, 0, offset);
    ASSERT(offset == RegExpMatchesIndicesGroupsPropertyOffset);
    return structure;
}

Structure* createRegExpMatchesArrayStructure(VM& vm, JSGlobalObject* globalObject)
{
    return createStructureImpl(vm, globalObject, ArrayWithContiguous);
}

Structure* createRegExpMatchesArrayWithIndicesStructure(VM& vm, JSGlobalObject* globalObject)
{
    return createStructureWithIndicesImpl(vm, globalObject, ArrayWithContiguous);
}

Structure* createRegExpMatchesIndicesArrayStructure(VM& vm, JSGlobalObject* globalObject)
{
    return createIndicesStructureImpl(vm, globalObject, ArrayWithContiguous);
}

Structure* createRegExpMatchesArraySlowPutStructure(VM& vm, JSGlobalObject* globalObject)
{
    return createStructureImpl(vm, globalObject, ArrayWithSlowPutArrayStorage);
}

Structure* createRegExpMatchesArrayWithIndicesSlowPutStructure(VM& vm, JSGlobalObject* globalObject)
{
    return createStructureWithIndicesImpl(vm, globalObject, ArrayWithSlowPutArrayStorage);
}

Structure* createRegExpMatchesIndicesArraySlowPutStructure(VM& vm, JSGlobalObject* globalObject)
{
    return createIndicesStructureImpl(vm, globalObject, ArrayWithSlowPutArrayStorage);
}

} // namespace JSC
