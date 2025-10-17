/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 21, 2021.
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

#include "Exception.h"
#include <wtf/CrossThreadCopier.h>
#include <wtf/Expected.h>
#include <wtf/StdLibExtras.h>

namespace WebCore {

template<typename T> class ExceptionOr {
public:
    using ReturnType = T;

    ExceptionOr(Exception&&);
    ExceptionOr(ReturnType&&);
    template<typename OtherType> ExceptionOr(const OtherType&, typename std::enable_if<std::is_scalar<OtherType>::value && std::is_convertible<OtherType, ReturnType>::value>::type* = nullptr);

    bool hasException() const;
    const Exception& exception() const;
    Exception releaseException();
    const ReturnType& returnValue() const;
    ReturnType releaseReturnValue();
    
private:
    Expected<ReturnType, Exception> m_value;
#if ASSERT_ENABLED
    bool m_wasReleased { false };
#endif
};

template<typename T> class ExceptionOr<T&> {
public:
    using ReturnType = T&;
    using ReturnReferenceType = T;

    ExceptionOr(Exception&&);
    ExceptionOr(ReturnReferenceType&);

    bool hasException() const;
    const Exception& exception() const;
    Exception releaseException();
    const ReturnReferenceType& returnValue() const;
    ReturnReferenceType& releaseReturnValue();
    
private:
    ExceptionOr<ReturnReferenceType*> m_value;
};

template<> class ExceptionOr<void> {
public:
    using ReturnType = void;

    ExceptionOr(Exception&&);
    ExceptionOr() = default;

    bool hasException() const;
    const Exception& exception() const;
    Exception releaseException();

private:
    Expected<void, Exception> m_value;
#if ASSERT_ENABLED
    bool m_wasReleased { false };
#endif
};

template<typename ReturnType> inline ExceptionOr<ReturnType>::ExceptionOr(Exception&& exception)
    : m_value(makeUnexpected(WTFMove(exception)))
{
}

template<typename ReturnType> inline ExceptionOr<ReturnType>::ExceptionOr(ReturnType&& returnValue)
    : m_value(WTFMove(returnValue))
{
}

template<typename ReturnType> template<typename OtherType> inline ExceptionOr<ReturnType>::ExceptionOr(const OtherType& returnValue, typename std::enable_if<std::is_scalar<OtherType>::value && std::is_convertible<OtherType, ReturnType>::value>::type*)
    : m_value(static_cast<ReturnType>(returnValue))
{
}

template<typename ReturnType> inline bool ExceptionOr<ReturnType>::hasException() const
{
    return !m_value.has_value();
}

template<typename ReturnType> inline const Exception& ExceptionOr<ReturnType>::exception() const
{
    ASSERT(!m_wasReleased);
    return m_value.error();
}

template<typename ReturnType> inline Exception ExceptionOr<ReturnType>::releaseException()
{
    ASSERT(!std::exchange(m_wasReleased, true));
    return WTFMove(m_value.error());
}

template<typename ReturnType> inline const ReturnType& ExceptionOr<ReturnType>::returnValue() const
{
    ASSERT(!m_wasReleased);
    return m_value.value();
}

template<typename ReturnType> inline ReturnType ExceptionOr<ReturnType>::releaseReturnValue()
{
    ASSERT(!std::exchange(m_wasReleased, true));
    return WTFMove(m_value.value());
}

template<typename ReturnReferenceType> inline ExceptionOr<ReturnReferenceType&>::ExceptionOr(Exception&& exception)
    : m_value(WTFMove(exception))
{
}

template<typename ReturnReferenceType> inline ExceptionOr<ReturnReferenceType&>::ExceptionOr(ReturnReferenceType& returnValue)
    : m_value(&returnValue)
{
}

template<typename ReturnReferenceType> inline bool ExceptionOr<ReturnReferenceType&>::hasException() const
{
    return m_value.hasException();
}

template<typename ReturnReferenceType> inline const Exception& ExceptionOr<ReturnReferenceType&>::exception() const
{
    return m_value.exception();
}

template<typename ReturnReferenceType> inline Exception ExceptionOr<ReturnReferenceType&>::releaseException()
{
    return m_value.releaseException();
}

template<typename ReturnReferenceType> inline const ReturnReferenceType& ExceptionOr<ReturnReferenceType&>::returnValue() const
{
    return *m_value.returnValue();
}

template<typename ReturnReferenceType> inline ReturnReferenceType& ExceptionOr<ReturnReferenceType&>::releaseReturnValue()
{
    return *m_value.releaseReturnValue();
}

inline ExceptionOr<void>::ExceptionOr(Exception&& exception)
    : m_value(makeUnexpected(WTFMove(exception)))
{
}

inline bool ExceptionOr<void>::hasException() const
{
    return !m_value.has_value();
}

inline const Exception& ExceptionOr<void>::exception() const
{
    ASSERT(!m_wasReleased);
    return m_value.error();
}

inline Exception ExceptionOr<void>::releaseException()
{
    ASSERT(!std::exchange(m_wasReleased, true));
    return WTFMove(m_value.error());
}

template <typename T> inline constexpr bool IsExceptionOr = WTF::IsTemplate<std::decay_t<T>, ExceptionOr>::value;

template <typename T, bool isExceptionOr = IsExceptionOr<T>> struct TypeOrExceptionOrUnderlyingTypeImpl;
template <typename T> struct TypeOrExceptionOrUnderlyingTypeImpl<T, true> { using Type = typename T::ReturnType; };
template <typename T> struct TypeOrExceptionOrUnderlyingTypeImpl<T, false> { using Type = T; };
template <typename T> using TypeOrExceptionOrUnderlyingType = typename TypeOrExceptionOrUnderlyingTypeImpl<T>::Type;

}

namespace WTF {
template<typename T> struct CrossThreadCopierBase<false, false, WebCore::ExceptionOr<T> > {
    using Type = WebCore::ExceptionOr<T>;
    static constexpr bool IsNeeded = true;
    static Type copy(const Type& source)
    {
        if (source.hasException())
            return crossThreadCopy(source.exception());
        return crossThreadCopy(source.returnValue());
    }
    static Type copy(Type&& source)
    {
        if (source.hasException())
            return crossThreadCopy(source.releaseException());
        return crossThreadCopy(source.releaseReturnValue());
    }
};

template<> struct CrossThreadCopierBase<false, false, WebCore::ExceptionOr<void> > {
    using Type = WebCore::ExceptionOr<void>;
    static constexpr bool IsNeeded = true;
    static Type copy(const Type& source)
    {
        if (source.hasException())
            return crossThreadCopy(source.exception());
        return { };
    }
    static Type copy(Type&& source)
    {
        if (source.hasException())
            return crossThreadCopy(source.releaseException());
        return { };
    }
};

}
