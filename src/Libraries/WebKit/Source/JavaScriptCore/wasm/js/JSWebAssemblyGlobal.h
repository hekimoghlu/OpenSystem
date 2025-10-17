/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 24, 2025.
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
#include "WasmGlobal.h"
#include "WasmLimits.h"
#include "WebAssemblyFunction.h"
#include "WebAssemblyWrapperFunction.h"
#include <wtf/Ref.h>

namespace JSC {

class JSWebAssemblyGlobal final : public JSNonFinalObject {
public:
    using Base = JSNonFinalObject;
    static constexpr DestructionMode needsDestruction = NeedsDestruction;
    static void destroy(JSCell*);

    template<typename CellType, SubspaceAccess mode>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        return vm.webAssemblyGlobalSpace<mode>();
    }

    static JSWebAssemblyGlobal* create(VM&, Structure*, Ref<Wasm::Global>&&);
    static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

    DECLARE_INFO;

    Wasm::Global* global() { return m_global.ptr(); }
    JSObject* type(JSGlobalObject*);

private:
    JSWebAssemblyGlobal(VM&, Structure*, Ref<Wasm::Global>&&);
    DECLARE_DEFAULT_FINISH_CREATION;
    DECLARE_VISIT_CHILDREN;

    Ref<Wasm::Global> m_global;
};

} // namespace JSC

#endif // ENABLE(WEBASSEMBLY)
