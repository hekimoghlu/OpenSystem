/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 24, 2022.
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

#include "ExceptionOr.h"
#include "JSDOMGlobalObject.h"
#include "JSDOMPromiseDeferred.h"
#include <wtf/Function.h>
#include <wtf/Vector.h>

namespace WebCore {

template<typename IDLType>
class DOMPromiseProxy {
    WTF_MAKE_TZONE_ALLOCATED_TEMPLATE(DOMPromiseProxy);
public:
    using Value = typename IDLType::StorageType;

    DOMPromiseProxy() = default;
    ~DOMPromiseProxy() = default;

    JSC::JSValue promise(JSC::JSGlobalObject&, JSDOMGlobalObject&);

    void clear();

    bool isFulfilled() const;

    void resolve(typename IDLType::StorageType);
    void resolveWithNewlyCreated(typename IDLType::StorageType);
    void reject(Exception, RejectAsHandled = RejectAsHandled::No);
    
private:
    JSC::JSValue resolvePromise(JSC::JSGlobalObject&, JSDOMGlobalObject&, const Function<void(DeferredPromise&)>&);

    std::optional<ExceptionOr<Value>> m_valueOrException;
    Vector<Ref<DeferredPromise>, 1> m_deferredPromises;
};

WTF_MAKE_TZONE_ALLOCATED_TEMPLATE_IMPL(template<typename IDLType>, DOMPromiseProxy<IDLType>);

template<>
class DOMPromiseProxy<IDLUndefined> {
    WTF_MAKE_TZONE_ALLOCATED_TEMPLATE(DOMPromiseProxy);
public:
    DOMPromiseProxy() = default;
    ~DOMPromiseProxy() = default;

    JSC::JSValue promise(JSC::JSGlobalObject&, JSDOMGlobalObject&);

    void clear();

    bool isFulfilled() const;

    void resolve();
    void reject(Exception, RejectAsHandled = RejectAsHandled::No);

private:
    std::optional<ExceptionOr<void>> m_valueOrException;
    Vector<Ref<DeferredPromise>, 1> m_deferredPromises;
};

// Instead of storing the value of the resolution directly, DOMPromiseProxyWithResolveCallback
// allows the owner to specify callback to be called when the resolved value is needed. This is
// needed to avoid reference cycles when the resolved value is the owner, such as is the case with
// FontFace and FontFaceSet.
template<typename IDLType>
class DOMPromiseProxyWithResolveCallback {
    WTF_MAKE_TZONE_ALLOCATED_TEMPLATE(DOMPromiseProxyWithResolveCallback);
public:
    using ResolveCallback = Function<typename IDLType::ParameterType()>;

    template <typename Class, typename BaseClass>
    DOMPromiseProxyWithResolveCallback(Class&, typename IDLType::ParameterType (BaseClass::*)());
    DOMPromiseProxyWithResolveCallback(ResolveCallback&&);
    ~DOMPromiseProxyWithResolveCallback() = default;

    JSC::JSValue promise(JSC::JSGlobalObject&, JSDOMGlobalObject&);

    void clear();

    bool isFulfilled() const;

    void resolve(typename IDLType::ParameterType);
    void resolveWithNewlyCreated(typename IDLType::ParameterType);
    void reject(Exception, RejectAsHandled = RejectAsHandled::No);
    
private:
    ResolveCallback m_resolveCallback;
    std::optional<ExceptionOr<void>> m_valueOrException;
    Vector<Ref<DeferredPromise>, 1> m_deferredPromises;
};

WTF_MAKE_TZONE_ALLOCATED_TEMPLATE_IMPL(template<typename IDLType>, DOMPromiseProxyWithResolveCallback<IDLType>);

// MARK: - DOMPromiseProxy<IDLType> generic implementation

template<typename IDLType>
inline JSC::JSValue DOMPromiseProxy<IDLType>::resolvePromise(JSC::JSGlobalObject& lexicalGlobalObject, JSDOMGlobalObject& globalObject, const Function<void(DeferredPromise&)>& resolvePromiseCallback)
{
    UNUSED_PARAM(lexicalGlobalObject);
    for (auto& deferredPromise : m_deferredPromises) {
        if (deferredPromise->globalObject() == &globalObject)
            return deferredPromise->promise();
    }

    // DeferredPromise can fail construction during worker abrupt termination.
    auto deferredPromise = DeferredPromise::create(globalObject, DeferredPromise::Mode::RetainPromiseOnResolve);
    if (!deferredPromise)
        return JSC::jsUndefined();

    m_deferredPromises.append(*deferredPromise);

    if (m_valueOrException) {
        // Calls to reject() / resolvePromiseCallback() may destroy |this|.
        if (m_valueOrException->hasException())
            deferredPromise->reject(m_valueOrException->exception());
        else
            resolvePromiseCallback(*deferredPromise);
    }

    return deferredPromise->promise();
}

template<typename IDLType>
inline JSC::JSValue DOMPromiseProxy<IDLType>::promise(JSC::JSGlobalObject& lexicalGlobalObject, JSDOMGlobalObject& globalObject)
{
    return resolvePromise(lexicalGlobalObject, globalObject, [this](auto& deferredPromise) {
        deferredPromise.template resolve<IDLType>(m_valueOrException->returnValue());
    });
}

template<>
inline JSC::JSValue DOMPromiseProxy<IDLAny>::promise(JSC::JSGlobalObject& lexicalGlobalObject, JSDOMGlobalObject& globalObject)
{
    return resolvePromise(lexicalGlobalObject, globalObject, [this](auto& deferredPromise) {
        deferredPromise.resolveWithJSValue(m_valueOrException->returnValue().get());
    });
}

template<typename IDLType>
inline void DOMPromiseProxy<IDLType>::clear()
{
    m_valueOrException = std::nullopt;
    m_deferredPromises.clear();
}

template<typename IDLType>
inline bool DOMPromiseProxy<IDLType>::isFulfilled() const
{
    return m_valueOrException.has_value();
}

template<typename IDLType>
inline void DOMPromiseProxy<IDLType>::resolve(typename IDLType::StorageType value)
{
    ASSERT(!m_valueOrException);

    m_valueOrException = ExceptionOr<Value> { std::forward<typename IDLType::StorageType>(value) };
    auto deferredPromisesCopy = m_deferredPromises;
    auto returnValueCopy = m_valueOrException->returnValue();
    for (auto& deferredPromise : deferredPromisesCopy)
        deferredPromise->template resolve<IDLType>(returnValueCopy);
}

template<>
inline void DOMPromiseProxy<IDLAny>::resolve(typename IDLAny::StorageType value)
{
    ASSERT(!m_valueOrException);

    m_valueOrException = ExceptionOr<Value> { std::forward<typename IDLAny::StorageType>(value) };
    auto deferredPromisesCopy = m_deferredPromises;
    auto returnValueCopy = m_valueOrException->returnValue();
    for (auto& deferredPromise : deferredPromisesCopy)
        deferredPromise->resolveWithJSValue(returnValueCopy.get());
}

template<typename IDLType>
inline void DOMPromiseProxy<IDLType>::resolveWithNewlyCreated(typename IDLType::StorageType value)
{
    ASSERT(!m_valueOrException);

    m_valueOrException = ExceptionOr<Value> { std::forward<typename IDLType::StorageType>(value) };
    auto deferredPromisesCopy = m_deferredPromises;
    auto returnValueCopy = m_valueOrException->returnValue();
    for (auto& deferredPromise : deferredPromisesCopy)
        deferredPromise->template resolveWithNewlyCreated<IDLType>(returnValueCopy);
}

template<typename IDLType>
inline void DOMPromiseProxy<IDLType>::reject(Exception exception, RejectAsHandled rejectAsHandled)
{
    ASSERT(!m_valueOrException);

    m_valueOrException = ExceptionOr<Value> { WTFMove(exception) };
    auto deferredPromisesCopy = m_deferredPromises;
    auto exceptionCopy = m_valueOrException->exception();
    for (auto& deferredPromise : deferredPromisesCopy)
        deferredPromise->reject(exceptionCopy, rejectAsHandled);
}


// MARK: - DOMPromiseProxy<IDLUndefined> specialization

inline JSC::JSValue DOMPromiseProxy<IDLUndefined>::promise(JSC::JSGlobalObject& lexicalGlobalObject, JSDOMGlobalObject& globalObject)
{
    UNUSED_PARAM(lexicalGlobalObject);
    for (auto& deferredPromise : m_deferredPromises) {
        if (deferredPromise->globalObject() == &globalObject)
            return deferredPromise->promise();
    }

    // DeferredPromise can fail construction during worker abrupt termination.
    auto deferredPromise = DeferredPromise::create(globalObject, DeferredPromise::Mode::RetainPromiseOnResolve);
    if (!deferredPromise)
        return JSC::jsUndefined();

    m_deferredPromises.append(*deferredPromise);

    if (m_valueOrException) {
        // Calls to reject() / resolve() may destroy |this|.
        if (m_valueOrException->hasException())
            deferredPromise->reject(m_valueOrException->exception());
        else
            deferredPromise->resolve();
    }

    return deferredPromise->promise();
}

inline void DOMPromiseProxy<IDLUndefined>::clear()
{
    m_valueOrException = std::nullopt;
    m_deferredPromises.clear();
}

inline bool DOMPromiseProxy<IDLUndefined>::isFulfilled() const
{
    return m_valueOrException.has_value();
}

inline void DOMPromiseProxy<IDLUndefined>::resolve()
{
    ASSERT(!m_valueOrException);
    m_valueOrException = ExceptionOr<void> { };
    auto deferredPromisesCopy = m_deferredPromises;
    for (auto& deferredPromise : deferredPromisesCopy)
        deferredPromise->resolve();
}

inline void DOMPromiseProxy<IDLUndefined>::reject(Exception exception, RejectAsHandled rejectAsHandled)
{
    ASSERT(!m_valueOrException);
    m_valueOrException = ExceptionOr<void> { WTFMove(exception) };
    auto deferredPromisesCopy = m_deferredPromises;
    auto exceptionCopy = m_valueOrException->exception();
    for (auto& deferredPromise : deferredPromisesCopy)
        deferredPromise->reject(exceptionCopy, rejectAsHandled);
}

// MARK: - DOMPromiseProxyWithResolveCallback<IDLType> implementation

template<typename IDLType>
template <typename Class, typename BaseClass>
inline DOMPromiseProxyWithResolveCallback<IDLType>::DOMPromiseProxyWithResolveCallback(Class& object, typename IDLType::ParameterType (BaseClass::*function)())
    : m_resolveCallback(std::bind(function, &object))
{
}

template<typename IDLType>
inline DOMPromiseProxyWithResolveCallback<IDLType>::DOMPromiseProxyWithResolveCallback(ResolveCallback&& function)
    : m_resolveCallback(WTFMove(function))
{
}

template<typename IDLType>
inline JSC::JSValue DOMPromiseProxyWithResolveCallback<IDLType>::promise(JSC::JSGlobalObject& lexicalGlobalObject, JSDOMGlobalObject& globalObject)
{
    UNUSED_PARAM(lexicalGlobalObject);
    for (auto& deferredPromise : m_deferredPromises) {
        if (deferredPromise->globalObject() == &globalObject)
            return deferredPromise->promise();
    }

    // DeferredPromise can fail construction during worker abrupt termination.
    auto deferredPromise = DeferredPromise::create(globalObject, DeferredPromise::Mode::RetainPromiseOnResolve);
    if (!deferredPromise)
        return JSC::jsUndefined();

    m_deferredPromises.append(*deferredPromise);

    if (m_valueOrException) {
        // Calls to reject() / resolve() may destroy |this|.
        if (m_valueOrException->hasException())
            deferredPromise->reject(m_valueOrException->exception());
        else
            deferredPromise->template resolve<IDLType>(m_resolveCallback());
    }

    return deferredPromise->promise();
}

template<typename IDLType>
inline void DOMPromiseProxyWithResolveCallback<IDLType>::clear()
{
    m_valueOrException = std::nullopt;
    m_deferredPromises.clear();
}

template<typename IDLType>
inline bool DOMPromiseProxyWithResolveCallback<IDLType>::isFulfilled() const
{
    return m_valueOrException.has_value();
}

template<typename IDLType>
inline void DOMPromiseProxyWithResolveCallback<IDLType>::resolve(typename IDLType::ParameterType value)
{
    ASSERT(!m_valueOrException);

    m_valueOrException = ExceptionOr<void> { };
    auto deferredPromisesCopy = m_deferredPromises;
    for (auto& deferredPromise : deferredPromisesCopy)
        deferredPromise->template resolve<IDLType>(value);
}

template<typename IDLType>
inline void DOMPromiseProxyWithResolveCallback<IDLType>::resolveWithNewlyCreated(typename IDLType::ParameterType value)
{
    ASSERT(!m_valueOrException);

    m_valueOrException = ExceptionOr<void> { };
    auto deferredPromisesCopy = m_deferredPromises;
    for (auto& deferredPromise : deferredPromisesCopy)
        deferredPromise->template resolveWithNewlyCreated<IDLType>(value);
}

template<typename IDLType>
inline void DOMPromiseProxyWithResolveCallback<IDLType>::reject(Exception exception, RejectAsHandled rejectAsHandled)
{
    ASSERT(!m_valueOrException);

    m_valueOrException = ExceptionOr<void> { WTFMove(exception) };
    auto deferredPromisesCopy = m_deferredPromises;
    auto exceptionCopy = m_valueOrException->exception();
    for (auto& deferredPromise : deferredPromisesCopy)
        deferredPromise->reject(exceptionCopy, rejectAsHandled);
}

}
