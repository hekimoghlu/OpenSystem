/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 20, 2023.
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
#include "WasmTag.h"

namespace JSC {

class JSWebAssemblyException final : public JSNonFinalObject {
public:
    using Base = JSNonFinalObject;
    static constexpr DestructionMode needsDestruction = NeedsDestruction;
    using Payload = FixedVector<uint64_t>;

    static void destroy(JSCell*);

    template<typename CellType, SubspaceAccess mode>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        return vm.webAssemblyExceptionSpace<mode>();
    }

    DECLARE_EXPORT_INFO;

    static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

    static JSWebAssemblyException* create(VM& vm, Structure* structure, Ref<const Wasm::Tag>&& tag, FixedVector<uint64_t>&& payload)
    {
        JSWebAssemblyException* exception = new (NotNull, allocateCell<JSWebAssemblyException>(vm)) JSWebAssemblyException(vm, structure, WTFMove(tag), WTFMove(payload));
        exception->finishCreation(vm);
        return exception;
    }

    DECLARE_VISIT_CHILDREN;

    const Wasm::Tag& tag() const { return m_tag.get(); }
    const FixedVector<uint64_t>& payload() const { return m_payload; }
    JSValue getArg(JSGlobalObject*, unsigned) const;

    static constexpr ptrdiff_t offsetOfPayload() { return OBJECT_OFFSETOF(JSWebAssemblyException, m_payload); }

protected:
    JSWebAssemblyException(VM&, Structure*, Ref<const Wasm::Tag>&&, FixedVector<uint64_t>&&);

    void finishCreation(VM&);

    Ref<const Wasm::Tag> m_tag;
    Payload m_payload;
};

} // namespace JSC

#endif // ENABLE(WEBASSEMBLY)
