/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 10, 2025.
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

#include "JSCJSValue.h"
#include <wtf/Assertions.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/VectorTraits.h>

namespace JSC {

    class CallFrame;
    class CodeBlock;
    class JSLexicalEnvironment;
    class JSObject;
    class JSScope;

    class Register {
        WTF_MAKE_TZONE_NON_HEAP_ALLOCATABLE(Register);
    public:
        Register();

        Register(const JSValue&);
        JSValue jsValue() const;
        JSValue asanUnsafeJSValue() const;
        EncodedJSValue encodedJSValue() const;
        
        ALWAYS_INLINE Register& operator=(CallFrame*);
        ALWAYS_INLINE Register& operator=(CodeBlock*);
        ALWAYS_INLINE Register& operator=(JSScope*);
        ALWAYS_INLINE Register& operator=(JSCell*);
        ALWAYS_INLINE Register& operator=(EncodedJSValue);

        int32_t i() const;
        ALWAYS_INLINE CallFrame* callFrame() const;
        ALWAYS_INLINE CodeBlock* codeBlock() const;
        ALWAYS_INLINE CodeBlock* asanUnsafeCodeBlock() const;
        ALWAYS_INLINE JSObject* object() const;
        ALWAYS_INLINE JSScope* scope() const;
        int32_t unboxedInt32() const;
        uint32_t unboxedUInt32() const;
        int32_t asanUnsafeUnboxedInt32() const;
        int64_t unboxedInt52() const;
        int64_t asanUnsafeUnboxedInt52() const;
        int64_t unboxedStrictInt52() const;
        int64_t asanUnsafeUnboxedStrictInt52() const;
        int64_t unboxedInt64() const;
        int64_t asanUnsafeUnboxedInt64() const;
        bool unboxedBoolean() const;
#if ENABLE(WEBASSEMBLY) && USE(JSVALUE32_64)
        float unboxedFloat() const;
        float asanUnsafeUnboxedFloat() const;
#endif
        double unboxedDouble() const;
        double asanUnsafeUnboxedDouble() const;
        JSCell* unboxedCell() const;
        JSCell* asanUnsafeUnboxedCell() const;
        int32_t payload() const;
        int32_t tag() const;
        int32_t unsafePayload() const;
        int32_t unsafeTag() const;
        int32_t& payload();
        int32_t& tag();

        void* pointer() const;
        void* asanUnsafePointer() const;

        static Register withInt(int32_t i)
        {
            Register r = jsNumber(i);
            return r;
        }

    private:
        union {
            EncodedJSValue value;
            CallFrame* callFrame;
            CodeBlock* codeBlock;
            EncodedValueDescriptor encodedValue;
            double number;
            int64_t integer;
        } u;
    };

    ALWAYS_INLINE Register::Register()
    {
#ifndef NDEBUG
        *this = JSValue();
#endif
    }

    ALWAYS_INLINE Register::Register(const JSValue& v)
    {
        u.value = JSValue::encode(v);
    }

    // FIXME (rdar://problem/19379214): ASan only needs to be suppressed for Register::jsValue() when called from prepareOSREntry(), but there is currently no way to express this short of adding a separate copy of the function.
    SUPPRESS_ASAN ALWAYS_INLINE JSValue Register::asanUnsafeJSValue() const
    {
        return JSValue::decode(u.value);
    }

    ALWAYS_INLINE JSValue Register::jsValue() const
    {
        return JSValue::decode(u.value);
    }

    ALWAYS_INLINE EncodedJSValue Register::encodedJSValue() const
    {
        return u.value;
    }

    // Interpreter functions

    ALWAYS_INLINE int32_t Register::i() const
    {
        return jsValue().asInt32();
    }

    ALWAYS_INLINE int32_t Register::unboxedInt32() const
    {
        return payload();
    }

    ALWAYS_INLINE uint32_t Register::unboxedUInt32() const
    {
        return static_cast<uint32_t>(unboxedInt32());
    }

    SUPPRESS_ASAN ALWAYS_INLINE int32_t Register::asanUnsafeUnboxedInt32() const
    {
        return unsafePayload();
    }

    ALWAYS_INLINE int64_t Register::unboxedInt52() const
    {
        return u.integer >> JSValue::int52ShiftAmount;
    }

    SUPPRESS_ASAN ALWAYS_INLINE int64_t Register::asanUnsafeUnboxedInt52() const
    {
        return u.integer >> JSValue::int52ShiftAmount;
    }

    ALWAYS_INLINE int64_t Register::unboxedStrictInt52() const
    {
        return u.integer;
    }

    SUPPRESS_ASAN ALWAYS_INLINE int64_t Register::asanUnsafeUnboxedStrictInt52() const
    {
        return u.integer;
    }

    ALWAYS_INLINE int64_t Register::unboxedInt64() const
    {
        return u.integer;
    }

    SUPPRESS_ASAN ALWAYS_INLINE int64_t Register::asanUnsafeUnboxedInt64() const
    {
        return u.integer;
    }

    ALWAYS_INLINE bool Register::unboxedBoolean() const
    {
        return !!payload();
    }

#if ENABLE(WEBASSEMBLY) && USE(JSVALUE32_64)
    ALWAYS_INLINE float Register::unboxedFloat() const
    {
        return std::bit_cast<float>(payload());
    }

    SUPPRESS_ASAN ALWAYS_INLINE float Register::asanUnsafeUnboxedFloat() const
    {
        return std::bit_cast<float>(payload());
    }
#endif

    ALWAYS_INLINE double Register::unboxedDouble() const
    {
        return u.number;
    }

    SUPPRESS_ASAN ALWAYS_INLINE double Register::asanUnsafeUnboxedDouble() const
    {
        return u.number;
    }

    ALWAYS_INLINE JSCell* Register::unboxedCell() const
    {
#if USE(JSVALUE64)
        return u.encodedValue.ptr;
#else
        return std::bit_cast<JSCell*>(payload());
#endif
    }

    SUPPRESS_ASAN ALWAYS_INLINE JSCell* Register::asanUnsafeUnboxedCell() const
    {
#if USE(JSVALUE64)
        return u.encodedValue.ptr;
#else
        return std::bit_cast<JSCell*>(payload());
#endif
    }

    ALWAYS_INLINE void* Register::pointer() const
    {
#if USE(JSVALUE64)
        return u.encodedValue.ptr;
#else
        return std::bit_cast<void*>(payload());
#endif
    }

    SUPPRESS_ASAN ALWAYS_INLINE void* Register::asanUnsafePointer() const
    {
#if USE(JSVALUE64)
        return u.encodedValue.ptr;
#else
        return std::bit_cast<void*>(unsafePayload());
#endif
    }

    ALWAYS_INLINE int32_t Register::payload() const
    {
        return u.encodedValue.asBits.payload;
    }

    ALWAYS_INLINE int32_t Register::tag() const
    {
        return u.encodedValue.asBits.tag;
    }

    SUPPRESS_ASAN ALWAYS_INLINE int32_t Register::unsafePayload() const
    {
        return u.encodedValue.asBits.payload;
    }

    SUPPRESS_ASAN ALWAYS_INLINE int32_t Register::unsafeTag() const
    {
        return u.encodedValue.asBits.tag;
    }

    ALWAYS_INLINE int32_t& Register::payload()
    {
        return u.encodedValue.asBits.payload;
    }

    ALWAYS_INLINE int32_t& Register::tag()
    {
        return u.encodedValue.asBits.tag;
    }

} // namespace JSC

namespace WTF {

    template<> struct VectorTraits<JSC::Register> : VectorTraitsBase<true, JSC::Register> { };

} // namespace WTF
