/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 4, 2023.
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

class AsyncGeneratorFunctionPrototype;

class AsyncGeneratorFunctionConstructor final : public InternalFunction {
public:
    using Base = InternalFunction;

    DECLARE_INFO;

    static AsyncGeneratorFunctionConstructor* create(VM& vm, Structure* structure, AsyncGeneratorFunctionPrototype* asyncGeneratorFunctionPrototype)
    {
        AsyncGeneratorFunctionConstructor* constructor = new (NotNull, allocateCell<AsyncGeneratorFunctionConstructor>(vm)) AsyncGeneratorFunctionConstructor(vm, structure);
        constructor->finishCreation(vm, asyncGeneratorFunctionPrototype);
        return constructor;
    }

    static Structure* createStructure(VM& vm, JSGlobalObject* globalObject, JSValue prototype)
    {
        return Structure::create(vm, globalObject, prototype, TypeInfo(InternalFunctionType, StructureFlags), info());
    }

private:
    AsyncGeneratorFunctionConstructor(VM&, Structure*);
    void finishCreation(VM&, AsyncGeneratorFunctionPrototype*);
};
STATIC_ASSERT_ISO_SUBSPACE_SHARABLE(AsyncGeneratorFunctionConstructor, InternalFunction);

} // namespace JSC
