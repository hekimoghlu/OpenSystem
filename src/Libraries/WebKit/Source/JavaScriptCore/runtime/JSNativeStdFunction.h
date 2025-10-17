/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 26, 2024.
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

class JSGlobalObject;

using NativeStdFunction = WTF::Function<EncodedJSValue(JSGlobalObject*, CallFrame*)>;

class JSNativeStdFunction final : public JSFunction {
public:
    using Base = JSFunction;

    static constexpr unsigned StructureFlags = Base::StructureFlags;
    static constexpr DestructionMode needsDestruction = NeedsDestruction;
    static void destroy(JSCell* cell)
    {
        static_cast<JSNativeStdFunction*>(cell)->JSNativeStdFunction::~JSNativeStdFunction();
    }

    template<typename CellType, SubspaceAccess mode>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        return vm.nativeStdFunctionSpace<mode>();
    }

    DECLARE_EXPORT_INFO;

    JS_EXPORT_PRIVATE static JSNativeStdFunction* create(VM&, JSGlobalObject*, unsigned length, const String& name, NativeStdFunction&&, Intrinsic = NoIntrinsic, NativeFunction nativeConstructor = callHostFunctionAsConstructor);

    inline static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

    const NativeStdFunction& function() { return m_function; }

private:
    JSNativeStdFunction(VM&, NativeExecutable*, JSGlobalObject*, Structure*, NativeStdFunction&&);
    void finishCreation(VM&, NativeExecutable*, unsigned length, const String& name);
    DECLARE_VISIT_CHILDREN;

    NativeStdFunction m_function;
};

} // namespace JSC
