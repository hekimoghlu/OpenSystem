/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 12, 2023.
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

class FinalizationRegistryPrototype;

class FinalizationRegistryConstructor final : public InternalFunction {
public:
    using Base = InternalFunction;

    static FinalizationRegistryConstructor* create(VM& vm, Structure* structure, FinalizationRegistryPrototype* prototype)
    {
        FinalizationRegistryConstructor* constructor = new (NotNull, allocateCell<FinalizationRegistryConstructor>(vm)) FinalizationRegistryConstructor(vm, structure);
        constructor->finishCreation(vm, prototype);
        return constructor;
    }

    DECLARE_INFO;

    inline static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

private:
    FinalizationRegistryConstructor(VM&, Structure*);
    void finishCreation(VM&, FinalizationRegistryPrototype*);
};

STATIC_ASSERT_ISO_SUBSPACE_SHARABLE(FinalizationRegistryConstructor, InternalFunction);

}
