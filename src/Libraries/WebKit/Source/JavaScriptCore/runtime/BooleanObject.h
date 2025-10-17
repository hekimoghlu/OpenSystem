/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 6, 2024.
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

class BooleanObject : public JSWrapperObject {
protected:
    JS_EXPORT_PRIVATE BooleanObject(VM&, Structure*);
    DECLARE_DEFAULT_FINISH_CREATION;

public:
    using Base = JSWrapperObject;

    template<typename, SubspaceAccess mode>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        return vm.booleanObjectSpace<mode>();
    }

    static BooleanObject* create(VM& vm, Structure* structure)
    {
        BooleanObject* boolean = new (NotNull, allocateCell<BooleanObject>(vm)) BooleanObject(vm, structure);
        boolean->finishCreation(vm);
        return boolean;
    }
        
    DECLARE_EXPORT_INFO;
        
    inline static Structure* createStructure(VM&, JSGlobalObject*, JSValue);
};
static_assert(sizeof(BooleanObject) == sizeof(JSWrapperObject));

} // namespace JSC
