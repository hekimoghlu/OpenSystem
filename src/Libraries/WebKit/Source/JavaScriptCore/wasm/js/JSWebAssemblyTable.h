/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 5, 2025.
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
#include "WasmLimits.h"
#include "WasmTable.h"
#include "WebAssemblyWrapperFunction.h"
#include "WebAssemblyFunction.h"
#include <wtf/Ref.h>

namespace JSC {

class JSWebAssemblyTable final : public JSNonFinalObject {
public:
    using Base = JSNonFinalObject;
    static constexpr DestructionMode needsDestruction = NeedsDestruction;
    static void destroy(JSCell*);

    template<typename CellType, SubspaceAccess mode>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        return vm.webAssemblyTableSpace<mode>();
    }

    static JSWebAssemblyTable* create(VM&, Structure*, Ref<Wasm::Table>&&);
    static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

    DECLARE_INFO;

    static bool isValidLength(uint32_t length) { return Wasm::Table::isValidLength(length); }
    std::optional<uint32_t> maximum() const { return m_table->maximum(); }
    uint32_t length() const { return m_table->length(); }
    uint32_t allocatedLength() const { return m_table->allocatedLength(length()); }
    std::optional<uint32_t> grow(JSGlobalObject*, uint32_t delta, JSValue defaultValue) WARN_UNUSED_RETURN;
    JSValue get(JSGlobalObject*, uint32_t);
    void set(uint32_t, JSValue);
    void set(JSGlobalObject*, uint32_t, JSValue);
    void clear(uint32_t);
    JSObject* type(JSGlobalObject*);

    Wasm::Table* table() { return m_table.ptr(); }

private:
    JSWebAssemblyTable(VM&, Structure*, Ref<Wasm::Table>&&);
    DECLARE_DEFAULT_FINISH_CREATION;
    DECLARE_VISIT_CHILDREN;

    Ref<Wasm::Table> m_table;
};

} // namespace JSC

#endif // ENABLE(WEBASSEMBLY)
