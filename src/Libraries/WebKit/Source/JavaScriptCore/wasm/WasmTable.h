/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 6, 2025.
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

#include "SlotVisitorMacros.h"
#include "WasmCallee.h"
#include "WasmFormat.h"
#include "WasmLimits.h"
#include "WriteBarrier.h"
#include <wtf/MallocPtr.h>
#include <wtf/Ref.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/ThreadSafeRefCounted.h>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace JSC {

class JSWebAssemblyTable;
class WebAssemblyFunctionBase;

namespace Wasm {

class FuncRefTable;

class Table : public ThreadSafeRefCounted<Table> {
    WTF_MAKE_NONCOPYABLE(Table);
    WTF_MAKE_TZONE_ALLOCATED(Table);
public:
    static RefPtr<Table> tryCreate(uint32_t initial, std::optional<uint32_t> maximum, TableElementType, Type);

    JS_EXPORT_PRIVATE ~Table() = default;

    std::optional<uint32_t> maximum() const { return m_maximum; }
    uint32_t length() const { return m_length; }

    static constexpr ptrdiff_t offsetOfLength() { return OBJECT_OFFSETOF(Table, m_length); }

    static uint32_t allocatedLength(uint32_t length);

    JSWebAssemblyTable* owner() const { return m_owner; }
    void setOwner(JSWebAssemblyTable* owner)
    {
        ASSERT(!m_owner);
        ASSERT(owner);
        m_owner = owner;
    }

    TableElementType type() const { return m_type; }
    bool isExternrefTable() const { return m_type == TableElementType::Externref; }
    bool isFuncrefTable() const { return m_type == TableElementType::Funcref; }
    Type wasmType() const { return m_wasmType; }
    FuncRefTable* asFuncrefTable();

    static bool isValidLength(uint32_t length) { return length < maxTableEntries; }

    void clear(uint32_t);
    void set(uint32_t, JSValue);
    JSValue get(uint32_t) const;

    std::optional<uint32_t> grow(uint32_t delta, JSValue defaultValue);
    void copy(const Table* srcTable, uint32_t dstIndex, uint32_t srcIndex);

    DECLARE_VISIT_AGGREGATE;

    void operator delete(Table*, std::destroying_delete_t);

protected:
    Table(uint32_t initial, std::optional<uint32_t> maximum, Type, TableElementType = TableElementType::Externref);

    template<typename Visitor> constexpr decltype(auto) visitDerived(Visitor&&);
    template<typename Visitor> constexpr decltype(auto) visitDerived(Visitor&&) const;

    void setLength(uint32_t);

    bool isFixedSized() const { return m_isFixedSized; }

    uint32_t m_length;
    NO_UNIQUE_ADDRESS const std::optional<uint32_t> m_maximum;
    const TableElementType m_type;
    Type m_wasmType;
    bool m_isFixedSized { false };
    JSWebAssemblyTable* m_owner;
};

class ExternOrAnyRefTable final : public Table {
    WTF_MAKE_TZONE_ALLOCATED(ExternOrAnyRefTable);
public:
    friend class Table;

    void clear(uint32_t);
    void set(uint32_t, JSValue);
    JSValue get(uint32_t index) const { return m_jsValues.get()[index].get(); }

private:
    ExternOrAnyRefTable(uint32_t initial, std::optional<uint32_t> maximum, Type wasmType);

    MallocPtr<WriteBarrier<Unknown>, VMMalloc> m_jsValues;
};

class FuncRefTable final : public Table {
    WTF_MAKE_TZONE_ALLOCATED(FuncRefTable);
public:
    friend class Table;

    JS_EXPORT_PRIVATE ~FuncRefTable();

    // call_indirect needs to do an Instance check to potentially context switch when calling a function to another instance. We can hold raw pointers to JSWebAssemblyInstance here because the js ensures that Table keeps all the instances alive.
    struct Function {
        WasmOrJSImportableFunction m_function;
        WasmOrJSImportableFunctionCallLinkInfo* m_callLinkInfo { nullptr };
        JSWebAssemblyInstance* m_instance { nullptr };
        WriteBarrier<Unknown> m_value { NullWriteBarrierTag };
        // In the case when we do not JIT, we cannot use the WasmToJSCallee singleton.
        // This callee gives the jitless wasm_to_js thunk the info it needs to call the imported
        // function with the correct wasm type.
        // Note that wasm to js calls will have m_function's boxedWasmCalleeLoadLocation already set.
        RefPtr<WasmToJSCallee> m_protectedJSCallee;

        static constexpr ptrdiff_t offsetOfFunction() { return OBJECT_OFFSETOF(Function, m_function); }
        static constexpr ptrdiff_t offsetOfCallLinkInfo() { return OBJECT_OFFSETOF(Function, m_callLinkInfo); }
        static constexpr ptrdiff_t offsetOfInstance() { return OBJECT_OFFSETOF(Function, m_instance); }
        static constexpr ptrdiff_t offsetOfValue() { return OBJECT_OFFSETOF(Function, m_value); }
    };

    void setFunction(uint32_t, WebAssemblyFunctionBase*);
    const Function& function(uint32_t) const;
    void copyFunction(const FuncRefTable* srcTable, uint32_t dstIndex, uint32_t srcIndex);

    static constexpr ptrdiff_t offsetOfFunctions() { return OBJECT_OFFSETOF(FuncRefTable, m_importableFunctions); }
    static constexpr ptrdiff_t offsetOfTail() { return WTF::roundUpToMultipleOf<alignof(Function)>(sizeof(FuncRefTable)); }
    static constexpr ptrdiff_t offsetOfFunctionsForFixedSizedTable() { return offsetOfTail(); }

    static size_t allocationSize(uint32_t size)
    {
        return offsetOfTail() + sizeof(Function) * size;
    }

    void clear(uint32_t);
    void set(uint32_t, JSValue);
    JSValue get(uint32_t index) const { return m_importableFunctions.get()[index].m_value.get(); }

private:
    FuncRefTable(uint32_t initial, std::optional<uint32_t> maximum, Type wasmType);

    Function* tailPointer() { return std::bit_cast<Function*>(std::bit_cast<uint8_t*>(this) + offsetOfTail()); }

    static Ref<FuncRefTable> createFixedSized(uint32_t size, Type wasmType);

    MallocPtr<Function, VMMalloc> m_importableFunctions;
};

} } // namespace JSC::Wasm

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

#endif // ENABLE(WEBASSEMBLY)
