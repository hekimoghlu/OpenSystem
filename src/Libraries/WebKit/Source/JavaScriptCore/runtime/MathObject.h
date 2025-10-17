/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 6, 2025.
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

#include "JSObject.h"

namespace JSC {

class MathObject final : public JSNonFinalObject {
public:
    using Base = JSNonFinalObject;

    template<typename CellType, SubspaceAccess>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        STATIC_ASSERT_ISO_SUBSPACE_SHARABLE(MathObject, Base);
        return &vm.plainObjectSpace();
    }

    static MathObject* create(VM& vm, JSGlobalObject* globalObject, Structure* structure)
    {
        MathObject* object = new (NotNull, allocateCell<MathObject>(vm)) MathObject(vm, structure);
        object->finishCreation(vm, globalObject);
        return object;
    }

    DECLARE_INFO;

    inline static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

private:
    MathObject(VM&, Structure*);
    void finishCreation(VM&, JSGlobalObject*);
};

JSC_DECLARE_HOST_FUNCTION(mathProtoFuncAbs);
JSC_DECLARE_HOST_FUNCTION(mathProtoFuncFloor);
JSC_DECLARE_HOST_FUNCTION(mathProtoFuncMin);

} // namespace JSC
