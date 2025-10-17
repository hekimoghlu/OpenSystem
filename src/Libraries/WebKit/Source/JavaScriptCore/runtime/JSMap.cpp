/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 5, 2025.
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
#include "JSMap.h"

#include "JSCInlines.h"

namespace JSC {

const ClassInfo JSMap::s_info = { "Map"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(JSMap) };

JSMap* JSMap::clone(JSGlobalObject* globalObject, VM& vm, Structure* structure)
{
    JSMap* instance = new (NotNull, allocateCell<JSMap>(vm)) JSMap(vm, structure);
    instance->finishCreation(globalObject, vm, this);
    return instance;
}

bool JSMap::isSetFastAndNonObservable(Structure* structure)
{
    JSGlobalObject* globalObject = structure->globalObject();
    if (!globalObject->isMapPrototypeSetFastAndNonObservable())
        return false;

    if (structure->hasPolyProto())
        return false;

    if (structure->storedPrototype() != globalObject->mapPrototype())
        return false;

    return true;
}

bool JSMap::isIteratorProtocolFastAndNonObservable()
{
    JSGlobalObject* globalObject = this->globalObject();
    if (!globalObject->isMapPrototypeIteratorProtocolFastAndNonObservable())
        return false;

    VM& vm = globalObject->vm();
    Structure* structure = this->structure();
    // This is the fast case. Many maps will be an original map.
    if (structure == globalObject->mapStructure())
        return true;

    if (getPrototypeDirect() != globalObject->mapPrototype())
        return false;

    if (getDirectOffset(vm, vm.propertyNames->iteratorSymbol) != invalidOffset)
        return false;

    return true;
}

}
