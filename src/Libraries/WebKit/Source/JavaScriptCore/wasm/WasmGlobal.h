/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 7, 2025.
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
#include "WasmFormat.h"
#include "WasmLimits.h"
#include "WriteBarrier.h"
#include <wtf/Ref.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/ThreadSafeRefCounted.h>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace JSC {

class JSWebAssemblyGlobal;

namespace Wasm {

class Global final : public ThreadSafeRefCounted<Global> {
    WTF_MAKE_NONCOPYABLE(Global);
    WTF_MAKE_TZONE_ALLOCATED(Global);
public:
    union Value {
        v128_t m_vector { };
        uint64_t m_primitive;
        WriteBarrierBase<Unknown> m_externref;
        Value* m_pointer;
    };
    static_assert(sizeof(Value) == 16, "Update LLInt if this changes");

    static Ref<Global> create(Wasm::Type type, Wasm::Mutability mutability, uint64_t initialValue = 0)
    {
        return adoptRef(*new Global(type, mutability, initialValue));
    }

    static Ref<Global> create(Wasm::Type type, Wasm::Mutability mutability, v128_t initialValue)
    {
        return adoptRef(*new Global(type, mutability, initialValue));
    }

    Wasm::Type type() const { return m_type; }
    Wasm::Mutability mutability() const { return m_mutability; }
    JSValue get(JSGlobalObject*) const;
    uint64_t getPrimitive() const { return m_value.m_primitive; }
    v128_t getVector() const { return m_value.m_vector; }
    void set(JSGlobalObject*, JSValue);
    DECLARE_VISIT_AGGREGATE;

    JSWebAssemblyGlobal* owner() const { return m_owner; }
    void setOwner(JSWebAssemblyGlobal* owner)
    {
        ASSERT(!m_owner);
        ASSERT(owner);
        m_owner = owner;
    }

    static constexpr ptrdiff_t offsetOfValue() { ASSERT(!OBJECT_OFFSETOF(Value, m_primitive)); ASSERT(!OBJECT_OFFSETOF(Value, m_externref)); return OBJECT_OFFSETOF(Global, m_value); }
    static constexpr ptrdiff_t offsetOfOwner() { return OBJECT_OFFSETOF(Global, m_owner); }

    static Global& fromBinding(Value& value)
    {
        return *std::bit_cast<Global*>(std::bit_cast<uint8_t*>(&value) - offsetOfValue());
    }

    Value* valuePointer() { return &m_value; }

private:
    Global(Wasm::Type type, Wasm::Mutability mutability, uint64_t initialValue)
        : m_type(type)
        , m_mutability(mutability)
    {
        ASSERT(m_type != Types::V128);
        m_value.m_primitive = initialValue;
    }

    Global(Wasm::Type type, Wasm::Mutability mutability, v128_t initialValue)
        : m_type(type)
        , m_mutability(mutability)
    {
        ASSERT(m_type == Types::V128);
        m_value.m_vector = initialValue;
    }

    Wasm::Type m_type;
    Wasm::Mutability m_mutability;
    JSWebAssemblyGlobal* m_owner { nullptr };
    Value m_value;
};

} } // namespace JSC::Wasm

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

#endif // ENABLE(WEBASSEMBLY)
