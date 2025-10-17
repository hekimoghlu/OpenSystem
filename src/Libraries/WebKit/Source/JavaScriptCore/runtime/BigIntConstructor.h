/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 26, 2023.
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

class BigIntPrototype;
class GetterSetter;

class BigIntConstructor final : public InternalFunction {
public:
    using Base = InternalFunction;
    static constexpr unsigned StructureFlags = Base::StructureFlags | HasStaticPropertyTable;

    static BigIntConstructor* create(VM& vm, Structure* structure, BigIntPrototype* bigIntPrototype)
    {
        BigIntConstructor* constructor = new (NotNull, allocateCell<BigIntConstructor>(vm)) BigIntConstructor(vm, structure);
        constructor->finishCreation(vm, bigIntPrototype);
        return constructor;
    }

    DECLARE_INFO;

    inline static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

private:
    BigIntConstructor(VM&, Structure*);
    void finishCreation(VM&, BigIntPrototype*);
};
STATIC_ASSERT_ISO_SUBSPACE_SHARABLE(BigIntConstructor, InternalFunction);

} // namespace JSC
