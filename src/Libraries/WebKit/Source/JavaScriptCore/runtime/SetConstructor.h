/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 18, 2024.
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

#include "InternalFunction.h"

namespace JSC {

class SetPrototype;
class GetterSetter;

class SetConstructor final : public InternalFunction {
public:
    typedef InternalFunction Base;

    static SetConstructor* create(VM& vm, Structure* structure, SetPrototype* setPrototype)
    {
        SetConstructor* constructor = new (NotNull, allocateCell<SetConstructor>(vm)) SetConstructor(vm, structure);
        constructor->finishCreation(vm, setPrototype);
        return constructor;
    }

    DECLARE_INFO;

    inline static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

private:
    SetConstructor(VM&, Structure*);
    void finishCreation(VM&, SetPrototype*);
};
STATIC_ASSERT_ISO_SUBSPACE_SHARABLE(SetConstructor, InternalFunction);

JSC_DECLARE_HOST_FUNCTION(setPrivateFuncSetStorage);
JSC_DECLARE_HOST_FUNCTION(setPrivateFuncSetIterationNext);
JSC_DECLARE_HOST_FUNCTION(setPrivateFuncSetIterationEntry);
JSC_DECLARE_HOST_FUNCTION(setPrivateFuncSetIterationEntryKey);
JSC_DECLARE_HOST_FUNCTION(setPrivateFuncClone);

} // namespace JSC
