/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 14, 2024.
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
#include <wtf/AccessibleAddress.h>
#include <wtf/StdLibExtras.h>

namespace JSC {

class JSCell;
class NativeCallee;

class CalleeBits {
public:
    CalleeBits() = default;
    CalleeBits(int64_t value)
#if USE(JSVALUE64)
        : m_ptr { reinterpret_cast<void*>(value) }
#elif USE(JSVALUE32_64)
        : m_ptr { reinterpret_cast<void*>(JSValue::decode(value).payload()) }
        , m_tag { JSValue::decode(value).tag() }
#endif
    { }
    CalleeBits(NativeCallee* nativeCallee) { *this = nativeCallee; }

    CalleeBits& operator=(JSCell* cell)
    {
        m_ptr = cell;
#if USE(JSVALUE32_64)
        m_tag = JSValue::CellTag;
#endif
        ASSERT(isCell());
        return *this;
    }

    CalleeBits& operator=(NativeCallee* nativeCallee)
    {
        m_ptr = boxNativeCalleeIfExists(nativeCallee);
#if USE(JSVALUE32_64)
        m_tag = JSValue::NativeCalleeTag;
#endif
        ASSERT_IMPLIES(nativeCallee, isNativeCallee());
        return *this;
    }

#if USE(JSVALUE32_64)
    static EncodedJSValue encodeNullCallee()
    {
        return JSValue::encode(jsNull());
    }

    static EncodedJSValue encodeJSCallee(const JSCell* cell)
    {
        if (!cell)
            return encodeNullCallee();
        return JSValue::encode(JSValue(cell));
    }

    static EncodedJSValue encodeBoxedNativeCallee(void* boxedCallee)
    {
        if (!boxedCallee)
            return encodeNullCallee();
        EncodedValueDescriptor ret;
        ret.asBits.tag = JSValue::NativeCalleeTag;
        ret.asBits.payload = reinterpret_cast<intptr_t>(boxedCallee);
        return std::bit_cast<EncodedJSValue>(ret);
    }

#elif USE(JSVALUE64)
    static EncodedJSValue encodeNullCallee()
    {
        return reinterpret_cast<EncodedJSValue>(nullptr);
    }

    static EncodedJSValue encodeJSCallee(const JSCell* cell)
    {
        if (!cell)
            return encodeNullCallee();
        return reinterpret_cast<EncodedJSValue>(cell);
    }

    static EncodedJSValue encodeBoxedNativeCallee(void* boxedCallee)
    {
        return reinterpret_cast<EncodedJSValue>(boxedCallee);
    }
#else
#error "Unsupported configuration"
#endif

    static EncodedJSValue encodeNativeCallee(NativeCallee* callee)
    {
        if (!callee)
            return encodeNullCallee();
        return encodeBoxedNativeCallee(boxNativeCallee(callee));
    }

    static void* boxNativeCalleeIfExists(NativeCallee* callee)
    {
        if (callee)
            return boxNativeCallee(callee);
        return nullptr;
    }

#if CPU(ARM64)
    // NativeCallees are sometimes stored in ThreadSafeWeakOrStrongPtr, which relies on top byte ignore, so we need to strip the top byte on ARM64.
    static constexpr uintptr_t nativeCalleeTopByteMask = std::numeric_limits<uintptr_t>::max() >> CHAR_BIT;
#endif

    static void* boxNativeCallee(NativeCallee* callee)
    {
#if USE(JSVALUE64)
        auto bits = std::bit_cast<uintptr_t>(callee);
#if CPU(ARM64)
        bits &= nativeCalleeTopByteMask;
#endif
        CalleeBits result { static_cast<int64_t>((bits - lowestAccessibleAddress()) | JSValue::NativeCalleeTag) };
        ASSERT(result.isNativeCallee());
        return result.rawPtr();
#elif USE(JSVALUE32_64)
        return std::bit_cast<void*>(std::bit_cast<uintptr_t>(callee) - lowestAccessibleAddress());
#endif
    }

    bool isNativeCallee() const
    {
#if USE(JSVALUE64)
        return (reinterpret_cast<uintptr_t>(m_ptr) & JSValue::NativeCalleeMask) == JSValue::NativeCalleeTag;
#elif USE(JSVALUE32_64)
        return m_tag == JSValue::NativeCalleeTag;
#endif
    }
    bool isCell() const { return !isNativeCallee(); }

    JSCell* asCell() const
    {
        ASSERT(!isNativeCallee());
        return static_cast<JSCell*>(m_ptr);
    }

    NativeCallee* asNativeCallee() const
    {
        ASSERT(isNativeCallee());
#if USE(JSVALUE64)
        return std::bit_cast<NativeCallee*>(static_cast<uintptr_t>(std::bit_cast<uintptr_t>(m_ptr) & ~JSValue::NativeCalleeTag) + lowestAccessibleAddress());
#elif USE(JSVALUE32_64)
        return std::bit_cast<NativeCallee*>(std::bit_cast<uintptr_t>(m_ptr) + lowestAccessibleAddress());
#endif
    }

    void* rawPtr() const { return m_ptr; }
    // For Ref/RefPtr support.
    explicit operator bool() const { return m_ptr; }

private:
    void* m_ptr { nullptr };
#if USE(JSVALUE32_64)
    uint32_t m_tag { JSValue::EmptyValueTag };
#endif
};

} // namespace JSC
