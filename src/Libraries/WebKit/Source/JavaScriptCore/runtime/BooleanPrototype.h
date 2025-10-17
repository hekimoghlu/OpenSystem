/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 21, 2023.
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

#include "BooleanObject.h"

namespace JSC {

class BooleanPrototype final : public BooleanObject {
public:
    using Base = BooleanObject;
    static constexpr unsigned StructureFlags = Base::StructureFlags | HasStaticPropertyTable;

    static BooleanPrototype* create(VM& vm, JSGlobalObject* globalObject, Structure* structure)
    {
        BooleanPrototype* prototype = new (NotNull, allocateCell<BooleanPrototype>(vm)) BooleanPrototype(vm, structure);
        prototype->finishCreation(vm, globalObject);
        return prototype;
    }
        
    DECLARE_INFO;

    inline static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

private:
    BooleanPrototype(VM&, Structure*);
    void finishCreation(VM&, JSGlobalObject*);
};
STATIC_ASSERT_ISO_SUBSPACE_SHARABLE(BooleanPrototype, BooleanObject);

} // namespace JSC
