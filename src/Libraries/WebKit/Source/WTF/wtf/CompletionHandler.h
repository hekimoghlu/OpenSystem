/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 25, 2023.
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

#include <tuple>
#include <utility>
#include <wtf/Function.h>
#include <wtf/MainThread.h>

namespace WTF {

template<typename> class CompletionHandler;
class CompletionHandlerCallThread {
public:
    static inline constexpr auto ConstructionThread = currentThreadLike;
    static inline constexpr auto MainThread = mainThreadLike;
    static inline constexpr auto AnyThread = anyThreadLike;
};

// Wraps a Function to make sure it is always called once and only once.
template <typename Out, typename... In>
class CompletionHandler<Out(In...)> {
    WTF_MAKE_FAST_ALLOCATED;
public:
    using OutType = Out;
    using InTypes = std::tuple<In...>;
    using Impl = typename Function<Out(In...)>::Impl;

    CompletionHandler() = default;

    template<typename CallableType, class = typename std::enable_if<std::is_rvalue_reference<CallableType&&>::value>::type>
    CompletionHandler(CallableType&& callable, ThreadLikeAssertion callThread = CompletionHandlerCallThread::ConstructionThread)
        : m_function(std::forward<CallableType>(callable))
        , m_callThread(WTFMove(callThread))
    {
    }

    CompletionHandler(CompletionHandler&&) = default;
    CompletionHandler& operator=(CompletionHandler&&) = default;

    ~CompletionHandler()
    {
        ASSERT_WITH_MESSAGE(!m_function, "Completion handler should always be called");
        m_callThread = anyThreadLike;
    }

    explicit operator bool() const { return !!m_function; }

    Impl* leak() { return m_function.leak(); }

    Out operator()(In... in)
    {
        assertIsCurrent(m_callThread);
        ASSERT_WITH_MESSAGE(m_function, "Completion handler should not be called more than once");
        return std::exchange(m_function, nullptr)(std::forward<In>(in)...);
    }

private:
    Function<Out(In...)> m_function;
    NO_UNIQUE_ADDRESS ThreadLikeAssertion m_callThread;
};

// Wraps a Function to make sure it is called at most once.
// If the CompletionHandlerWithFinalizer is destroyed and the function hasn't yet been called,
// the finalizer is invoked with the function as its argument.
template<typename> class CompletionHandlerWithFinalizer;
template <typename Out, typename... In>
class CompletionHandlerWithFinalizer<Out(In...)> {
    WTF_MAKE_FAST_ALLOCATED;
public:
    using OutType = Out;
    using InTypes = std::tuple<In...>;

    template<typename CallableType, class = typename std::enable_if<std::is_rvalue_reference<CallableType&&>::value>::type>
    CompletionHandlerWithFinalizer(CallableType&& callable, Function<void(Function<Out(In...)>&)>&& finalizer, ThreadLikeAssertion callThread = CompletionHandlerCallThread::ConstructionThread)
        : m_function(std::forward<CallableType>(callable))
        , m_finalizer(WTFMove(finalizer))
        , m_callThread(callThread)
    {
    }

    CompletionHandlerWithFinalizer(CompletionHandlerWithFinalizer&&) = default;
    CompletionHandlerWithFinalizer& operator=(CompletionHandlerWithFinalizer&&) = default;

    ~CompletionHandlerWithFinalizer()
    {
        if (!m_function)
            return;
        assertIsCurrent(m_callThread);
        m_finalizer(m_function);
    }

    explicit operator bool() const { return !!m_function; }

    Out operator()(In... in)
    {
        assertIsCurrent(m_callThread);
        ASSERT_WITH_MESSAGE(m_function, "Completion handler should not be called more than once");
        return std::exchange(m_function, nullptr)(std::forward<In>(in)...);
    }

private:
    Function<Out(In...)> m_function;
    Function<void(Function<Out(In...)>&)> m_finalizer;
    NO_UNIQUE_ADDRESS ThreadLikeAssertion m_callThread;
};

namespace Detail {

template<typename Out, typename... In>
class CallableWrapper<CompletionHandler<Out(In...)>, Out, In...> : public CallableWrapperBase<Out, In...> {
    WTF_MAKE_FAST_ALLOCATED;
public:
    explicit CallableWrapper(CompletionHandler<Out(In...)>&& completionHandler)
        : m_completionHandler(WTFMove(completionHandler))
    {
        RELEASE_ASSERT(m_completionHandler);
    }
    Out call(In... in) final { return m_completionHandler(std::forward<In>(in)...); }
private:
    CompletionHandler<Out(In...)> m_completionHandler;
};

} // namespace Detail

class CompletionHandlerCallingScope final {
    WTF_MAKE_FAST_ALLOCATED;
public:
    CompletionHandlerCallingScope() = default;

    CompletionHandlerCallingScope(CompletionHandler<void()>&& completionHandler)
        : m_completionHandler(WTFMove(completionHandler))
    { }

    ~CompletionHandlerCallingScope()
    {
        if (m_completionHandler)
            m_completionHandler();
    }

    CompletionHandlerCallingScope(CompletionHandlerCallingScope&&) = default;
    CompletionHandlerCallingScope& operator=(CompletionHandlerCallingScope&&) = default;

    CompletionHandler<void()> release() { return WTFMove(m_completionHandler); }

private:
    CompletionHandler<void()> m_completionHandler;
};

template<typename Out, typename... In> CompletionHandler<Out(In...)> adopt(typename CompletionHandler<Out(In...)>::Impl* impl)
{
    return Function<Out(In...)>(impl, Function<Out(In...)>::Adopt);
}

} // namespace WTF

using WTF::CompletionHandler;
using WTF::CompletionHandlerCallThread;
using WTF::CompletionHandlerCallingScope;
using WTF::CompletionHandlerWithFinalizer;
