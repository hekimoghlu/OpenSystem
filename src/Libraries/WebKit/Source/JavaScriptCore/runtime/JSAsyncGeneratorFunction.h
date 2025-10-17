/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 14, 2024.
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

#include "JSFunction.h"

namespace JSC {

class JSAsyncGeneratorFunction final : public JSFunction {
    friend class JIT;
    friend class VM;
public:
    using Base = JSFunction;

    static constexpr unsigned StructureFlags = Base::StructureFlags;

    DECLARE_EXPORT_INFO;

    static JSAsyncGeneratorFunction* create(VM&, JSGlobalObject*, FunctionExecutable*, JSScope*);
    static JSAsyncGeneratorFunction* create(VM&, JSGlobalObject*, FunctionExecutable*, JSScope*, Structure*);
    static JSAsyncGeneratorFunction* createWithInvalidatedReallocationWatchpoint(VM&, JSGlobalObject*, FunctionExecutable*, JSScope*);
    static JSAsyncGeneratorFunction* createWithInvalidatedReallocationWatchpoint(VM&, JSGlobalObject*, FunctionExecutable*, JSScope*, Structure*);

    static size_t allocationSize(size_t inlineCapacity)
    {
        ASSERT_UNUSED(inlineCapacity, !inlineCapacity);
        return sizeof(JSAsyncGeneratorFunction);
    }

    inline static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

private:
    JSAsyncGeneratorFunction(VM&, FunctionExecutable*, JSScope*, Structure*);

    static JSAsyncGeneratorFunction* createImpl(VM&, FunctionExecutable*, JSScope*, Structure*);
};
static_assert(sizeof(JSAsyncGeneratorFunction) == sizeof(JSFunction), "Some subclasses of JSFunction should be the same size to share IsoSubspace");

} // namespace JSC
