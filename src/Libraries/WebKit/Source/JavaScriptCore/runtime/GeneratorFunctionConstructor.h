/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 17, 2023.
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

namespace WTF {
class TextPosition;
}

namespace JSC {

class GeneratorFunctionPrototype;

// %GeneratorFunction% intrinsic.
// https://tc39.github.io/ecma262/#sec-generatorfunction-constructor
class GeneratorFunctionConstructor final : public InternalFunction {
public:
    typedef InternalFunction Base;

    static GeneratorFunctionConstructor* create(VM& vm, Structure* structure, GeneratorFunctionPrototype* generatorFunctionPrototype)
    {
        GeneratorFunctionConstructor* constructor = new (NotNull, allocateCell<GeneratorFunctionConstructor>(vm)) GeneratorFunctionConstructor(vm, structure);
        constructor->finishCreation(vm, generatorFunctionPrototype);
        return constructor;
    }

    DECLARE_INFO;

    inline static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

private:
    GeneratorFunctionConstructor(VM&, Structure*);
    void finishCreation(VM&, GeneratorFunctionPrototype*);
};
STATIC_ASSERT_ISO_SUBSPACE_SHARABLE(GeneratorFunctionConstructor, InternalFunction);

} // namespace JSC
