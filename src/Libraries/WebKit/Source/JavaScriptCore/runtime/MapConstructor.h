/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 4, 2022.
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

class MapPrototype;
class GetterSetter;

class MapConstructor final : public InternalFunction {
public:
    typedef InternalFunction Base;

    static MapConstructor* create(VM& vm, Structure* structure, MapPrototype* mapPrototype)
    {
        MapConstructor* constructor = new (NotNull, allocateCell<MapConstructor>(vm)) MapConstructor(vm, structure);
        constructor->finishCreation(vm, mapPrototype);
        return constructor;
    }

    DECLARE_INFO;

    inline static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

private:
    MapConstructor(VM&, Structure*);

    void finishCreation(VM&, MapPrototype*);
};
STATIC_ASSERT_ISO_SUBSPACE_SHARABLE(MapConstructor, InternalFunction);

JSC_DECLARE_HOST_FUNCTION(mapPrivateFuncMapIterationNext);
JSC_DECLARE_HOST_FUNCTION(mapPrivateFuncMapIterationEntry);
JSC_DECLARE_HOST_FUNCTION(mapPrivateFuncMapIterationEntryKey);
JSC_DECLARE_HOST_FUNCTION(mapPrivateFuncMapIterationEntryValue);
JSC_DECLARE_HOST_FUNCTION(mapPrivateFuncMapStorage);

} // namespace JSC
