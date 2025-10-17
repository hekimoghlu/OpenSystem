/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 23, 2024.
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
#include "WasmTypeDefinitionInlines.h"
#include "WebAssemblyGCObjectBase.h"
#include <wtf/Ref.h>

namespace JSC {

class JSWebAssemblyInstance;

class JSWebAssemblyStruct final : public WebAssemblyGCObjectBase {
public:
    using Base = WebAssemblyGCObjectBase;
    static constexpr DestructionMode needsDestruction = NeedsDestruction;

    static void destroy(JSCell*);

    template<typename CellType, SubspaceAccess mode>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        return vm.webAssemblyStructSpace<mode>();
    }

    DECLARE_EXPORT_INFO;

    static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

    static JSWebAssemblyStruct* create(VM&, Structure*, JSWebAssemblyInstance*, uint32_t, RefPtr<const Wasm::RTT>&&);

    DECLARE_VISIT_CHILDREN;

    uint64_t get(uint32_t) const;
    void set(uint32_t, uint64_t);
    void set(uint32_t, v128_t);
    const Wasm::StructType* structType() const { return m_type->as<Wasm::StructType>(); }
    Wasm::FieldType fieldType(uint32_t fieldIndex) const { return structType()->field(fieldIndex); }

    // Returns the offset for m_payload.m_storage
    static constexpr ptrdiff_t offsetOfPayload() { return OBJECT_OFFSETOF(JSWebAssemblyStruct, m_payload) + FixedVector<uint8_t>::offsetOfStorage(); }

    const uint8_t* fieldPointer(uint32_t fieldIndex) const;
    uint8_t* fieldPointer(uint32_t fieldIndex);

protected:
    JSWebAssemblyStruct(VM&, Structure*, Ref<const Wasm::TypeDefinition>&&, RefPtr<const Wasm::RTT>&&);
    DECLARE_DEFAULT_FINISH_CREATION;

    // FIXME: It is possible to encode the type information in the structure field of Wasm.Struct and remove this field.
    // https://bugs.webkit.org/show_bug.cgi?id=244838
    Ref<const Wasm::TypeDefinition> m_type;

    FixedVector<uint8_t> m_payload;
};

} // namespace JSC

#endif // ENABLE(WEBASSEMBLY)
