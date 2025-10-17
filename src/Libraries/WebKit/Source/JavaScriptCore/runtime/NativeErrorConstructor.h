/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 3, 2022.
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

#include "Error.h"
#include "InternalFunction.h"
#include "NativeErrorPrototype.h"

namespace JSC {

class ErrorInstance;
class FunctionPrototype;
class NativeErrorPrototype;

class NativeErrorConstructorBase : public InternalFunction {
public:
    using Base = InternalFunction;

    DECLARE_INFO;

    inline static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

protected:
    NativeErrorConstructorBase(VM& vm, Structure* structure, NativeFunction functionForCall, NativeFunction functionForConstruct)
        : InternalFunction(vm, structure, functionForCall, functionForConstruct)
    {
    }

    void finishCreation(VM&, NativeErrorPrototype*, ErrorType);
};

template<ErrorType errorType>
class NativeErrorConstructor final : public NativeErrorConstructorBase {
public:
    static NativeErrorConstructor* create(VM& vm, Structure* structure, NativeErrorPrototype* prototype)
    {
        NativeErrorConstructor* constructor = new (NotNull, allocateCell<NativeErrorConstructor>(vm)) NativeErrorConstructor(vm, structure);
        constructor->finishCreation(vm, prototype, errorType);
        return constructor;
    }

    static EncodedJSValue callImpl(JSGlobalObject*, CallFrame*);
    static EncodedJSValue constructImpl(JSGlobalObject*, CallFrame*);
private:
    NativeErrorConstructor(VM&, Structure*);
};

using EvalErrorConstructor = NativeErrorConstructor<ErrorType::EvalError>;
using RangeErrorConstructor = NativeErrorConstructor<ErrorType::RangeError>;
using ReferenceErrorConstructor = NativeErrorConstructor<ErrorType::ReferenceError>;
using SyntaxErrorConstructor = NativeErrorConstructor<ErrorType::SyntaxError>;
using TypeErrorConstructor = NativeErrorConstructor<ErrorType::TypeError>;
using URIErrorConstructor = NativeErrorConstructor<ErrorType::URIError>;

STATIC_ASSERT_ISO_SUBSPACE_SHARABLE(EvalErrorConstructor, InternalFunction);
STATIC_ASSERT_ISO_SUBSPACE_SHARABLE(RangeErrorConstructor, InternalFunction);
STATIC_ASSERT_ISO_SUBSPACE_SHARABLE(ReferenceErrorConstructor, InternalFunction);
STATIC_ASSERT_ISO_SUBSPACE_SHARABLE(SyntaxErrorConstructor, InternalFunction);
STATIC_ASSERT_ISO_SUBSPACE_SHARABLE(TypeErrorConstructor, InternalFunction);
STATIC_ASSERT_ISO_SUBSPACE_SHARABLE(URIErrorConstructor, InternalFunction);

} // namespace JSC
