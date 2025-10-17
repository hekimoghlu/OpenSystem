/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 6, 2024.
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

#include <JavaScriptCore/ExceptionScope.h>
#include <functional>
#include <type_traits>
#include <utility>
#include <wtf/Expected.h>

namespace WebCore {

template<typename T> struct Converter;

// Result a conversion from JSValue -> Implementation.
template<typename IDL> class ConversionResult;

// Token used to indicate that a conversion from JSValue -> Implementation has failed.
struct ConversionResultException { };

namespace Detail {

template<typename T>
struct ConversionResultStorage {
    using ReturnType = T;
    using Type = T;

    ConversionResultStorage(ConversionResultException token) : value(makeUnexpected(token)) { }
    ConversionResultStorage(const Type& value) : value(value) { }
    ConversionResultStorage(Type&& value) : value(WTFMove(value)) { }

    template<typename U>
    ConversionResultStorage(ConversionResultStorage<U>&& other)
        : value([&]() -> Expected<Type, ConversionResultException> {
            if (other.hasException())
                return makeUnexpected(ConversionResultException());
            return ReturnType { other.releaseReturnValue() };
        }())
    {
    }

    // Special case conversion from T& to T*
    template<typename U>
        requires (std::is_pointer_v<Type> && std::is_lvalue_reference_v<U>)
    ConversionResultStorage(ConversionResultStorage<U>&& other)
        : value([&]() -> Expected<Type, ConversionResultException> {
            if (other.hasException())
                return makeUnexpected(ConversionResultException());
            return ReturnType { &other.releaseReturnValue() };
        }())
    {
    }

    bool hasException() const
    {
        return !value.has_value();
    }

    ReturnType& returnValue()
    {
        ASSERT(!wasReleased);
        return value.value();
    }

    const ReturnType& returnValue() const
    {
        ASSERT(!wasReleased);
        return value.value();
    }

    ReturnType releaseReturnValue()
    {
        ASSERT(!std::exchange(wasReleased, true));
        return WTFMove(value.value());
    }

    Expected<Type, ConversionResultException> value;
#if ASSERT_ENABLED
    bool wasReleased { false };
#endif
};

template<typename T>
struct ConversionResultStorage<T&> {
    using ReturnType = T&;
    using Type = T;

    ConversionResultStorage(ConversionResultException token) : value(makeUnexpected(token)) { }
    ConversionResultStorage(Type& value) : value(std::reference_wrapper<Type> { value }) { }

    template<typename U>
    ConversionResultStorage(ConversionResultStorage<U>&& other)
        : value([&]() -> Expected<Type, ConversionResultException> {
            if (other.hasException())
                return makeUnexpected(ConversionResultException());
            return static_cast<WebCore::Detail::ConversionResultStorage<T&>::ReturnType>(other.releaseReturnValue());
        }())
    {
    }

    bool hasException() const
    {
        return !value.has_value();
    }

    Type& returnValue()
    {
        ASSERT(!wasReleased);
        return value.value().get();
    }

    const Type& returnValue() const
    {
        ASSERT(!wasReleased);
        return value.value().get();
    }

    Type& releaseReturnValue()
    {
        ASSERT(!std::exchange(wasReleased, true));
        return WTFMove(value.value()).get();
    }

    Expected<std::reference_wrapper<Type>, ConversionResultException> value;
#if ASSERT_ENABLED
    bool wasReleased { false };
#endif
};

} // namespace Detail

template<typename IDL>
class ConversionResult {
public:
    using ReturnType = typename Converter<IDL>::ReturnType;

    static ConversionResult exception() { return ConversionResult(ConversionResultException()); }

    // Token type for indicating an exception has been thrown.
    ConversionResult(ConversionResultException token)
        : m_storage { token }
    {
    }

    ConversionResult(const ReturnType& returnValue)
        : m_storage { returnValue }
    {
    }

    ConversionResult(ReturnType&& returnValue) requires (!std::is_lvalue_reference_v<ReturnType>)
        : m_storage { WTFMove(returnValue) }
    {
    }

    ConversionResult(std::nullptr_t) requires std::is_same_v<decltype(IDL::nullValue()), std::nullptr_t>
        : m_storage { nullptr }
    {
    }

    template<typename OtherIDL>
    ConversionResult(ConversionResult<OtherIDL>&& other)
        : m_storage { WTFMove(other.m_storage) }
    {
    }

    bool hasException(JSC::ExceptionScope& scope) const
    {
        EXCEPTION_ASSERT(!!scope.exception() == scope.vm().traps().needHandling(JSC::VMTraps::NeedExceptionHandling));

#if ENABLE(EXCEPTION_SCOPE_VERIFICATION)
        if (m_storage.hasException()) {
            EXCEPTION_ASSERT(scope.vm().traps().maybeNeedHandling() && scope.vm().hasExceptionsAfterHandlingTraps());
            return true;
        }
        return false;
#else
        UNUSED_PARAM(scope);
        return m_storage.hasException();
#endif
    }

    decltype(auto) returnValue() { ASSERT(!m_storage.hasException()); return m_storage.returnValue(); }
    decltype(auto) returnValue() const { ASSERT(!m_storage.hasException()); return m_storage.returnValue(); }
    decltype(auto) releaseReturnValue() { ASSERT(!m_storage.hasException()); return m_storage.releaseReturnValue(); }

private:
    template<typename> friend class ConversionResult;

    Detail::ConversionResultStorage<ReturnType> m_storage;
};

} // namespace WebCore
