/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 29, 2025.
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

class JSAsyncFunction final : public JSFunction {
    friend class JIT;
    friend class VM;
public:
    typedef JSFunction Base;

    static constexpr unsigned StructureFlags = Base::StructureFlags;

    DECLARE_EXPORT_INFO;

    static JSAsyncFunction* create(VM&, JSGlobalObject*, FunctionExecutable*, JSScope*);
    static JSAsyncFunction* create(VM&, JSGlobalObject*, FunctionExecutable*, JSScope*, Structure*);
    static JSAsyncFunction* createWithInvalidatedReallocationWatchpoint(VM&, JSGlobalObject*, FunctionExecutable*, JSScope*);
    static JSAsyncFunction* createWithInvalidatedReallocationWatchpoint(VM&, JSGlobalObject*, FunctionExecutable*, JSScope*, Structure*);

    static size_t allocationSize(Checked<size_t> inlineCapacity)
    {
        ASSERT_UNUSED(inlineCapacity, !inlineCapacity);
        return sizeof(JSAsyncFunction);
    }

    inline static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

private:
    JSAsyncFunction(VM&, FunctionExecutable*, JSScope*, Structure*);

    static JSAsyncFunction* createImpl(VM&, FunctionExecutable*, JSScope*, Structure*);
};
static_assert(sizeof(JSAsyncFunction) == sizeof(JSFunction), "Some subclasses of JSFunction should be the same size to share IsoSubspace");

} // namespace JSC
