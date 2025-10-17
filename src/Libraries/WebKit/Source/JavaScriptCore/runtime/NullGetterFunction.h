/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 6, 2025.
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

class NullGetterFunction final : public InternalFunction {
public:
    typedef InternalFunction Base;

    static NullGetterFunction* create(VM& vm, Structure* structure)
    {
        // Since NullGetterFunction is per JSGlobalObject, we use put-without-transition in InternalFunction::finishCreation.
        NullGetterFunction* function = new (NotNull, allocateCell< NullGetterFunction>(vm))  NullGetterFunction(vm, structure);
        function->finishCreation(vm, 0, String(), PropertyAdditionMode::WithoutStructureTransition);
        return function;
    }

    DECLARE_EXPORT_INFO;

    inline static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

private:
    NullGetterFunction(VM&, Structure*);
};
STATIC_ASSERT_ISO_SUBSPACE_SHARABLE(NullGetterFunction, InternalFunction);

} // namespace JSC
