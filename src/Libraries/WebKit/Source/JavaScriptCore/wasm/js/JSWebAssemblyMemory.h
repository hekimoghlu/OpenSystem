/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 23, 2023.
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
#include "WasmMemory.h"
#include <wtf/Ref.h>
#include <wtf/RefPtr.h>

namespace JSC {

class ArrayBuffer;
class JSArrayBuffer;

// FIXME: Merge Wasm::Memory into this now that JSWebAssemblyInstance is the only instance object.
class JSWebAssemblyMemory final : public JSNonFinalObject {
public:
    using Base = JSNonFinalObject;
    static constexpr DestructionMode needsDestruction = NeedsDestruction;
    static void destroy(JSCell*);

    template<typename CellType, SubspaceAccess mode>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        return vm.webAssemblyMemorySpace<mode>();
    }

    JS_EXPORT_PRIVATE static JSWebAssemblyMemory* create(VM&, Structure*);
    static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

    DECLARE_EXPORT_INFO;

    JS_EXPORT_PRIVATE void adopt(Ref<Wasm::Memory>&&);
    Wasm::Memory& memory() { return m_memory.get(); }
    JSArrayBuffer* buffer(JSGlobalObject*);
    PageCount grow(VM&, JSGlobalObject*, uint32_t delta);
    JS_EXPORT_PRIVATE void growSuccessCallback(VM&, PageCount oldPageCount, PageCount newPageCount);

    JSObject* type(JSGlobalObject*);

    MemoryMode mode() const { return m_memory->mode(); }
    MemorySharingMode sharingMode() const { return m_memory->sharingMode(); }
    size_t mappedCapacity() const { return m_memory->mappedCapacity(); }
    void* basePointer() const { return m_memory->basePointer(); }

    static constexpr ptrdiff_t offsetOfMemory() { return OBJECT_OFFSETOF(JSWebAssemblyMemory, m_memory); }

private:
    JSWebAssemblyMemory(VM&, Structure*);
    void finishCreation(VM&);
    DECLARE_VISIT_CHILDREN;

    Ref<Wasm::Memory> m_memory;
    WriteBarrier<JSArrayBuffer> m_bufferWrapper;
    RefPtr<ArrayBuffer> m_buffer;
};

} // namespace JSC

#endif // ENABLE(WEBASSEMBLY)
