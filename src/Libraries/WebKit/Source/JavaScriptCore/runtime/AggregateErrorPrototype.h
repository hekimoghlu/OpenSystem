/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 4, 2022.
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

#include "ErrorPrototype.h"

namespace JSC {

class AggregateErrorPrototype final : public ErrorPrototypeBase {
public:
    using Base = ErrorPrototypeBase;

    template<typename CellType, SubspaceAccess>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        STATIC_ASSERT_ISO_SUBSPACE_SHARABLE(AggregateErrorPrototype, Base);
        return &vm.plainObjectSpace();
    }

    inline static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

    static AggregateErrorPrototype* create(VM& vm, Structure* structure)
    {
        AggregateErrorPrototype* prototype = new (NotNull, allocateCell<AggregateErrorPrototype>(vm)) AggregateErrorPrototype(vm, structure);
        prototype->finishCreation(vm);
        return prototype;
    }

private:
    AggregateErrorPrototype(VM&, Structure*);
    void finishCreation(VM&);
};

} // namespace JSC
