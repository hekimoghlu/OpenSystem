/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 20, 2022.
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

#include "JSWrapperObject.h"

namespace JSC {

class NumberObject : public JSWrapperObject {
protected:
    NumberObject(VM&, Structure*);
#if ASSERT_ENABLED
    void finishCreation(VM&);
#endif

public:
    using Base = JSWrapperObject;

    template<typename, SubspaceAccess mode>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        return &vm.numberObjectSpace();
    }

    static NumberObject* create(VM& vm, Structure* structure)
    {
        NumberObject* number = new (NotNull, allocateCell<NumberObject>(vm)) NumberObject(vm, structure);
        number->finishCreation(vm);
        return number;
    }

    DECLARE_EXPORT_INFO;

    static Structure* createStructure(VM& vm, JSGlobalObject* globalObject, JSValue prototype)
    {
        return Structure::create(vm, globalObject, prototype, TypeInfo(NumberObjectType, StructureFlags), info());
    }
};
static_assert(sizeof(NumberObject) == sizeof(JSWrapperObject));

JS_EXPORT_PRIVATE NumberObject* constructNumber(JSGlobalObject*, JSValue);

} // namespace JSC
