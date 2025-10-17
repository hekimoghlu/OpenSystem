/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 7, 2025.
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

#if ENABLE(WEBASSEMBLY)

#include "JSObject.h"
#include "JSPromise.h"

namespace JSC {

class JSWebAssembly final : public JSNonFinalObject {
public:
    using Base = JSNonFinalObject;
    static constexpr unsigned StructureFlags = Base::StructureFlags | HasStaticPropertyTable;

    template<typename CellType, SubspaceAccess>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        STATIC_ASSERT_ISO_SUBSPACE_SHARABLE(JSWebAssembly, Base);
        return &vm.plainObjectSpace();
    }

    static JSWebAssembly* create(VM&, JSGlobalObject*, Structure*);
    static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

    DECLARE_INFO;

    JS_EXPORT_PRIVATE static void webAssemblyModuleValidateAsync(JSGlobalObject*, JSPromise*, Vector<uint8_t>&&);
    static JSValue instantiate(JSGlobalObject*, JSPromise*, const Identifier&, JSValue);

    static void instantiateForStreaming(VM&, JSGlobalObject*, JSPromise*, JSWebAssemblyModule*, JSObject*);

private:
    JSWebAssembly(VM&, Structure*);
    void finishCreation(VM&, JSGlobalObject*);
};

JSC_DECLARE_HOST_FUNCTION(webAssemblyCompileStreamingInternal);
JSC_DECLARE_HOST_FUNCTION(webAssemblyInstantiateStreamingInternal);

} // namespace JSC

#endif // ENABLE(WEBASSEMBLY)
