/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 21, 2022.
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

class FunctionPrototype;
enum class SourceTaintedOrigin : uint8_t;

class FunctionConstructor final : public InternalFunction {
public:
    typedef InternalFunction Base;

    static FunctionConstructor* create(VM& vm, Structure* structure, FunctionPrototype* functionPrototype)
    {
        FunctionConstructor* constructor = new (NotNull, allocateCell<FunctionConstructor>(vm)) FunctionConstructor(vm, structure);
        constructor->finishCreation(vm, functionPrototype);
        return constructor;
    }

    DECLARE_INFO;

    inline static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

private:
    FunctionConstructor(VM&, Structure*);
    void finishCreation(VM&, FunctionPrototype*);
};
STATIC_ASSERT_ISO_SUBSPACE_SHARABLE(FunctionConstructor, InternalFunction);

enum class FunctionConstructionMode {
    Function,
    Generator,
    Async,
    AsyncGenerator,
};

JSObject* constructFunction(JSGlobalObject*, const ArgList&, const Identifier& functionName, const SourceOrigin&, const String& sourceURL, SourceTaintedOrigin, const WTF::TextPosition&, FunctionConstructionMode = FunctionConstructionMode::Function, JSValue newTarget = JSValue());
JSObject* constructFunction(JSGlobalObject*, CallFrame*, const ArgList&, FunctionConstructionMode = FunctionConstructionMode::Function, JSValue newTarget = JSValue());

JS_EXPORT_PRIVATE JSObject* constructFunctionSkippingEvalEnabledCheck(
    JSGlobalObject*, String&& program, LexicallyScopedFeatures, const Identifier&, const SourceOrigin&,
    const String&, SourceTaintedOrigin, const WTF::TextPosition&, int overrideLineNumber = -1,
    std::optional<int> functionConstructorParametersEndPosition = std::nullopt,
    FunctionConstructionMode = FunctionConstructionMode::Function, JSValue newTarget = JSValue());

} // namespace JSC
