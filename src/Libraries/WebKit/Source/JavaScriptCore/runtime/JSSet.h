/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 31, 2022.
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

#include "OrderedHashTable.h"

namespace JSC {

class JSSet final : public OrderedHashSet {
    using Base = OrderedHashSet;
public:

    DECLARE_EXPORT_INFO;

    template<typename CellType, SubspaceAccess mode>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        return vm.setSpace<mode>();
    }

    static size_t allocationSize(Checked<size_t> inlineCapacity)
    {
        ASSERT_UNUSED(inlineCapacity, !inlineCapacity);
        return sizeof(JSSet);
    }

    inline static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

    static JSSet* create(VM& vm, Structure* structure)
    {
        JSSet* instance = new (NotNull, allocateCell<JSSet>(vm)) JSSet(vm, structure);
        instance->finishCreation(vm);
        return instance;
    }

    static bool isAddFastAndNonObservable(Structure*);
    bool isIteratorProtocolFastAndNonObservable();
    JSSet* clone(JSGlobalObject*, VM&, Structure*);

private:
    JSSet(VM& vm, Structure* structure)
        : Base(vm, structure)
    {
    }
};

static_assert(std::is_final<JSSet>::value, "Required for JSType based casting");

} // namespace JSC
