/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 5, 2023.
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

namespace WTF {

namespace Detail {

template<typename Out, typename... In>
class CallableWrapperBase {
    WTF_MAKE_FAST_ALLOCATED;
public:
    virtual ~CallableWrapperBase() { }
    virtual Out call(In...) = 0;
};

template<typename, typename, typename...> class CallableWrapper;

template<typename CallableType, typename Out, typename... In>
class CallableWrapper : public CallableWrapperBase<Out, In...> {
public:
    explicit CallableWrapper(CallableType&& callable)
        : m_callable(WTFMove(callable)) { }
    CallableWrapper(const CallableWrapper&) = delete;
    CallableWrapper& operator=(const CallableWrapper&) = delete;
    Out call(In... in) final { return m_callable(std::forward<In>(in)...); }
private:
    CallableType m_callable;
};

} // namespace Detail

template<typename Out, typename... In> Function<Out(In...)> adopt(Detail::CallableWrapperBase<Out, In...>*);

template <typename Out, typename... In>
class Function<Out(In...)> {
    WTF_MAKE_FAST_ALLOCATED;
public:
    using Impl = Detail::CallableWrapperBase<Out, In...>;

    Function() = default;
    Function(std::nullptr_t) { }

    template<typename CallableType, class = typename std::enable_if<!(std::is_pointer<CallableType>::value && std::is_function<typename std::remove_pointer<CallableType>::type>::value) && std::is_rvalue_reference<CallableType&&>::value>::type>
    Function(CallableType&& callable)
        : m_callableWrapper(makeUnique<Detail::CallableWrapper<CallableType, Out, In...>>(std::forward<CallableType>(callable))) { }

    template<typename FunctionType, class = typename std::enable_if<std::is_pointer<FunctionType>::value && std::is_function<typename std::remove_pointer<FunctionType>::type>::value>::type>
    Function(FunctionType f)
        : m_callableWrapper(makeUnique<Detail::CallableWrapper<FunctionType, Out, In...>>(std::forward<FunctionType>(f))) { }

    Out operator()(In... in) const
    {
        ASSERT(m_callableWrapper);
        return m_callableWrapper->call(std::forward<In>(in)...);
    }

    explicit operator bool() const { return !!m_callableWrapper; }

    template<typename CallableType, class = typename std::enable_if<!(std::is_pointer<CallableType>::value && std::is_function<typename std::remove_pointer<CallableType>::type>::value) && std::is_rvalue_reference<CallableType&&>::value>::type>
    Function& operator=(CallableType&& callable)
    {
        m_callableWrapper = makeUnique<Detail::CallableWrapper<CallableType, Out, In...>>(std::forward<CallableType>(callable));
        return *this;
    }

    template<typename FunctionType, class = typename std::enable_if<std::is_pointer<FunctionType>::value && std::is_function<typename std::remove_pointer<FunctionType>::type>::value>::type>
    Function& operator=(FunctionType f)
    {
        m_callableWrapper = makeUnique<Detail::CallableWrapper<FunctionType, Out, In...>>(std::forward<FunctionType>(f));
        return *this;
    }

    Function& operator=(std::nullptr_t)
    {
        m_callableWrapper = nullptr;
        return *this;
    }

    Impl* leak()
    {
        return m_callableWrapper.release();
    }

private:
    enum AdoptTag { Adopt };
    Function(Impl* impl, AdoptTag)
        : m_callableWrapper(impl)
    {
    }

    friend Function adopt<Out, In...>(Impl*);

    std::unique_ptr<Impl> m_callableWrapper;
};

template<typename Out, typename... In> Function<Out(In...)> adopt(Detail::CallableWrapperBase<Out, In...>* impl)
{
    return Function<Out(In...)>(impl, Function<Out(In...)>::Adopt);
}

} // namespace WTF
