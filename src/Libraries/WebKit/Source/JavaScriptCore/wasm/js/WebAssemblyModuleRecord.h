/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 12, 2024.
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

#include "AbstractModuleRecord.h"
#include "WasmCreationMode.h"
#include "WasmModuleInformation.h"

namespace JSC {

class JSWebAssemblyInstance;
class JSWebAssemblyModule;
class WebAssemblyFunction;

// Based on the WebAssembly.Instance specification
// https://github.com/WebAssembly/design/blob/master/JS.md#webassemblyinstance-constructor
class WebAssemblyModuleRecord final : public AbstractModuleRecord {
    friend class LLIntOffsetsExtractor;
public:
    using Base = AbstractModuleRecord;

    static constexpr DestructionMode needsDestruction = NeedsDestruction;
    static void destroy(JSCell*);

    template<typename CellType, SubspaceAccess mode>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        return vm.webAssemblyModuleRecordSpace<mode>();
    }

    DECLARE_EXPORT_INFO;

    static Structure* createStructure(VM&, JSGlobalObject*, JSValue);
    static WebAssemblyModuleRecord* create(JSGlobalObject*, VM&, Structure*, const Identifier&, const Wasm::ModuleInformation&);

    void prepareLink(VM&, JSWebAssemblyInstance*);
    Synchronousness link(JSGlobalObject*, JSValue scriptFetcher);
    void initializeImports(JSGlobalObject*, JSObject* importObject, Wasm::CreationMode);
    void initializeExports(JSGlobalObject*);
    JS_EXPORT_PRIVATE JSValue evaluate(JSGlobalObject*);

    JSObject* exportsObject() const { return m_exportsObject.get(); }

    static constexpr ptrdiff_t offsetOfExportsObject() { return OBJECT_OFFSETOF(WebAssemblyModuleRecord, m_exportsObject); }

private:
    WebAssemblyModuleRecord(VM&, Structure*, const Identifier&);

    void finishCreation(JSGlobalObject*, VM&, const Wasm::ModuleInformation&);
    JSValue evaluateConstantExpression(JSGlobalObject*, const Vector<uint8_t>&, const Wasm::ModuleInformation&, Wasm::Type, uint64_t&);

    DECLARE_VISIT_CHILDREN;

    WriteBarrier<JSWebAssemblyInstance> m_instance;
    WriteBarrier<JSObject> m_startFunction;
    WriteBarrier<JSObject> m_exportsObject;
};

} // namespace JSC

#endif // ENABLE(WEBASSEMBLY)
