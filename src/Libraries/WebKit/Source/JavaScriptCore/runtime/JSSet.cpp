/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 24, 2024.
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
#include "JSSet.h"

#include "JSCInlines.h"

namespace JSC {

const ClassInfo JSSet::s_info = { "Set"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(JSSet) };

JSSet* JSSet::clone(JSGlobalObject* globalObject, VM& vm, Structure* structure)
{
    JSSet* instance = new (NotNull, allocateCell<JSSet>(vm)) JSSet(vm, structure);
    instance->finishCreation(globalObject, vm, this);
    return instance;
}

bool JSSet::isAddFastAndNonObservable(Structure* structure)
{
    JSGlobalObject* globalObject = structure->globalObject();
    if (!globalObject->isSetPrototypeAddFastAndNonObservable())
        return false;

    if (structure->hasPolyProto())
        return false;

    if (structure->storedPrototype() != globalObject->jsSetPrototype())
        return false;

    return true;
}

bool JSSet::isIteratorProtocolFastAndNonObservable()
{
    JSGlobalObject* globalObject = this->globalObject();
    if (!globalObject->isSetPrototypeIteratorProtocolFastAndNonObservable())
        return false;

    VM& vm = globalObject->vm();
    Structure* structure = this->structure();
    // This is the fast case. Many sets will be an original set.
    if (structure == globalObject->setStructure())
        return true;

    if (getPrototypeDirect() != globalObject->jsSetPrototype())
        return false;

    if (getDirectOffset(vm, vm.propertyNames->iteratorSymbol) != invalidOffset)
        return false;

    return true;
}

}
