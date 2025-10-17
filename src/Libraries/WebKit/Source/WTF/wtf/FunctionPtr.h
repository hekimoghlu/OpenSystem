/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 23, 2025.
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

#include <memory>
#include <wtf/FastMalloc.h>
#include <wtf/Forward.h>
#include <wtf/FunctionTraits.h>
#include <wtf/Hasher.h>
#include <wtf/PtrTag.h>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace WTF {

// FunctionAttributes and FunctionCallConvention are only needed because x86 builds
// (especially for Windows) supports different calling conventions. We need to
// preserve these attributes so that when the function pointer is used in a
// call, the C/C++ compiler will be able to use the matching calling convention.

enum class FunctionAttributes {
    None,
    JITOperation, // JIT_OPERATION_ATTRIBUTES
    JSCHostCall, // JSC_HOST_CALL_ATTRIBUTES
};

template<PtrTag tag, typename, FunctionAttributes = FunctionAttributes::None> class FunctionPtr;

template<FunctionAttributes, typename> struct FunctionCallConvention;

template<typename Out, typename... In>
struct FunctionCallConvention<FunctionAttributes::None, Out(In...)> {
    using Type = Out(*)(In...);
};

template<typename Out, typename... In>
struct FunctionCallConvention<FunctionAttributes::JITOperation, Out(In...)> {
    using Type = Out(JIT_OPERATION_ATTRIBUTES *)(In...);
};

template<typename Out, typename... In>
struct FunctionCallConvention<FunctionAttributes::JSCHostCall, Out(In...)> {
    using Type = Out(JSC_HOST_CALL_ATTRIBUTES *)(In...);
};


class FunctionPtrBase {
public:
    // We need to declare this in this non-template base. Otherwise, every use of
    // AlreadyTaggedValueTag will require a specialized template qualification.
    enum AlreadyTaggedValueTag { AlreadyTaggedValue };

    friend bool operator==(FunctionPtrBase, FunctionPtrBase) = default;
};

template<PtrTag tag, typename Out, typename... In, FunctionAttributes attr>
class FunctionPtr<tag, Out(In...), attr> : public FunctionPtrBase {
public:
    using Ptr = typename FunctionCallConvention<attr, Out(In...)>::Type;

    constexpr FunctionPtr() : m_ptr(nullptr) { }
    constexpr FunctionPtr(std::nullptr_t) : m_ptr(nullptr) { }

    constexpr FunctionPtr(Out(*ptr)(In...))
        : m_ptr(encode(ptr))
    { }

#if OS(WINDOWS)
    constexpr FunctionPtr(Out(SYSV_ABI *ptr)(In...))
        : m_ptr(encode(ptr))
    { }
#endif

// MSVC doesn't seem to treat functions with different calling conventions as
// different types; these methods already defined for fastcall, below.
#if CALLING_CONVENTION_IS_STDCALL && !OS(WINDOWS)
    constexpr FunctionPtr(Out(CDECL *ptr)(In...))
        : m_ptr(encode(ptr))
    { }
#endif

#if COMPILER_SUPPORTS(FASTCALL_CALLING_CONVENTION)
    constexpr FunctionPtr(Out(FASTCALL *ptr)(In...))
        : m_ptr(encode(ptr))
    { }
#endif

    Out operator()(In... in) const
    {
        ASSERT(m_ptr);
        return (*get())(std::forward<In>(in)...);
    }

    constexpr Ptr get() const { return decode(m_ptr); }

    template<PtrTag otherTag>
    FunctionPtr<otherTag, Out(In...), attr> retagged() const
    {
        static_assert(tag != otherTag);
        return FunctionPtr<otherTag, Out(In...), attr>(AlreadyTaggedValue, retaggedPtr<otherTag>());
    }

    constexpr void* taggedPtr() const { return reinterpret_cast<void*>(m_ptr); }

    template<PtrTag newTag>
    void* retaggedPtr() const
    {
        static_assert(tag != newTag);
        return retagCodePtr<void*, tag, newTag>(m_ptr);
    }

    void* untaggedPtr() const { return untagCodePtr<void*, tag>(m_ptr); }

    explicit operator bool() const { return !!m_ptr; }
    bool operator!() const { return !m_ptr; }

    friend bool operator==(FunctionPtr, FunctionPtr) = default;

    FunctionPtr& operator=(Ptr ptr)
    {
        m_ptr = encode(ptr);
        return *this;
    }

    template<PtrTag otherTag>
    FunctionPtr& operator=(const FunctionPtr<otherTag, Out(In...), attr>& other)
    {
        m_ptr = encode(other.get());
        return *this;
    }

    FunctionPtr& operator=(std::nullptr_t)
    {
        m_ptr = nullptr;
        return *this;
    }

protected:
    FunctionPtr(AlreadyTaggedValueTag, void* ptr)
        : m_ptr(std::bit_cast<Ptr>(ptr))
    {
        assertIsNullOrTaggedWith<tag>(ptr);
    }

    ALWAYS_INLINE static constexpr Ptr encode(Ptr ptr)
    {
        // Note: we cannot do the assert before this check because it will disqualify
        // this function for use in an constexpr context for some use cases.
        if constexpr (tag == CFunctionPtrTag)
            return ptr;
        assertIsNullOrCFunctionPtr(ptr);
        return retagCodePtr<CFunctionPtrTag, tag>(ptr);
    }

    ALWAYS_INLINE static constexpr Ptr decode(Ptr ptr)
    {
        if constexpr (tag == CFunctionPtrTag)
            return ptr;
        auto result = retagCodePtr<tag, CFunctionPtrTag>(ptr);
        assertIsNullOrCFunctionPtr(result);
        return result;
    }

    Ptr m_ptr;

    template<PtrTag, typename, FunctionAttributes> friend class FunctionPtr;
};

static_assert(sizeof(FunctionPtr<CFunctionPtrTag, void()>) == sizeof(void*));
#if COMPILER_SUPPORTS(BUILTIN_IS_TRIVIALLY_COPYABLE)
static_assert(__is_trivially_copyable(FunctionPtr<CFunctionPtrTag, void()>));
#endif

template<PtrTag tag, typename Out, typename... In, FunctionAttributes attr>
struct FunctionTraits<FunctionPtr<tag, Out(In...), attr>> : public FunctionTraits<Out(In...)> {
};

template<PtrTag tag, typename Out, typename... In, FunctionAttributes attr>
void add(Hasher& hasher, const FunctionPtr<tag, Out(In...), attr>& ptr)
{
    add(hasher, ptr.taggedPtr());
}

} // namespace WTF

using WTF::FunctionAttributes;
using WTF::FunctionPtr;

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
