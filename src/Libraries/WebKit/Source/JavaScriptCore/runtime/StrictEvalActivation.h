/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 12, 2022.
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

#include "JSScope.h"

namespace JSC {

class StrictEvalActivation final : public JSScope {
public:
    using Base = JSScope;

    template<typename CellType, SubspaceAccess mode>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        return vm.strictEvalActivationSpace<mode>();
    }

    static StrictEvalActivation* create(VM& vm, Structure* structure, JSScope* currentScope)
    {
        StrictEvalActivation* scope = new (NotNull, allocateCell<StrictEvalActivation>(vm)) StrictEvalActivation(vm, structure, currentScope);
        scope->finishCreation(vm);
        return scope;
    }

    static bool deleteProperty(JSCell*, JSGlobalObject*, PropertyName, DeletePropertySlot&);

    inline static Structure* createStructure(VM&, JSGlobalObject*, JSValue);
    
    DECLARE_INFO;

private:
    StrictEvalActivation(VM&, Structure*, JSScope*);
};

} // namespace JSC
