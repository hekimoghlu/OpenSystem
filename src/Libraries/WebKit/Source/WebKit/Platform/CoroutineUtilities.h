/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 30, 2023.
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

#include <concepts>
#include <wtf/CompletionHandler.h>
#include <wtf/TZoneMallocInlines.h>

#if __has_include(<coroutine>)
#include <coroutine>
#else
// FIXME: Remove this once all supported toolchains have non-experimental coroutine support.
#include <experimental/coroutine>
namespace std {
using std::experimental::coroutine_handle;
using std::experimental::suspend_never;
using std::experimental::suspend_always;
}
#endif

namespace WebKit {

template<typename PromiseType>
class CoroutineHandle {
public:
    CoroutineHandle() = default;
    CoroutineHandle(std::coroutine_handle<PromiseType>&& handle)
        : m_handle(std::exchange(handle, nullptr)) { }
    CoroutineHandle(CoroutineHandle&& other)
        : m_handle(std::exchange(other.m_handle, nullptr)) { }
    CoroutineHandle& operator=(CoroutineHandle&& other)
    {
        m_handle = std::exchange(other.m_handle, nullptr);
        return *this;
    }
    CoroutineHandle(const CoroutineHandle&) = delete;
    CoroutineHandle& operator=(const CoroutineHandle&) = delete;
    ~CoroutineHandle()
    {
        if (m_handle)
            m_handle.destroy();
    }

    std::coroutine_handle<PromiseType> handle() const { return m_handle; }
private:
    std::coroutine_handle<PromiseType> m_handle;
};

// Name based on https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2018/p1056r1.html
template<typename T>
class Lazy {
public:
    class PromiseBase {
    public:
        struct final_awaitable {
            bool await_ready() const noexcept { return false; }
            template<typename Promise>
            std::coroutine_handle<> await_suspend(std::coroutine_handle<Promise> coroutine) noexcept { return coroutine.promise().handle(); }
            void await_resume() noexcept { }
        };
        std::suspend_always initial_suspend() { return { }; }
        final_awaitable final_suspend() noexcept { return { }; }
        template<typename Promise>
        void unhandled_exception() { }
        void setHandle(std::coroutine_handle<> handle) { m_handle = handle; }
        std::coroutine_handle<> handle() { return m_handle; }
    private:
        std::coroutine_handle<> m_handle;
    };
    template<typename U>
    class Promise final : public PromiseBase {
    public:
        Lazy<U> get_return_object() { return Lazy<U> { std::coroutine_handle<Promise>::from_promise(*this) }; }
        void return_value(U&& value) { m_value = WTFMove(value); }
        U result() && { return std::exchange(m_value, { }); }
    private:
        U m_value;
    };
    template<std::same_as<void> U>
    class Promise<U> : public PromiseBase {
    public:
        Lazy<void> get_return_object() { return Lazy<void> { std::coroutine_handle<Promise>::from_promise(*this) }; }
        void return_void() { }
        void result() { }
    };
    using promise_type = Promise<T>;

    class Awaitable {
    public:
        Awaitable(std::coroutine_handle<promise_type> coroutine)
            : m_coroutine(coroutine) { }
        bool await_ready() const { return m_coroutine.done(); }
        std::coroutine_handle<> await_suspend(std::coroutine_handle<> handle)
        {
            m_coroutine.promise().setHandle(handle);
            return m_coroutine;
        }
        T await_resume() { return WTFMove(this->m_coroutine.promise()).result(); }
    private:
        std::coroutine_handle<promise_type> m_coroutine;
    };
    Lazy(std::coroutine_handle<promise_type> coroutine)
        : m_coroutine(std::exchange(coroutine, nullptr)) { }
    Awaitable operator co_await() const { return { m_coroutine.handle() }; }
private:
    CoroutineHandle<promise_type> m_coroutine;
};

struct Task {
    struct promise_type {
        Task get_return_object() { return { std::coroutine_handle<promise_type>::from_promise(*this) }; }
        std::suspend_never initial_suspend() { return { }; }
        std::suspend_always final_suspend() noexcept { return { }; }
        void unhandled_exception() { }
        void return_void() { }
    };
    std::coroutine_handle<promise_type> handle;
};

class CoroutineCaller {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(CoroutineCaller);
public:
    CoroutineCaller() = default;
    void setCoroutine(Function<Task()>&& coroutine)
    {
        m_coroutine = std::exchange(coroutine, { });
        m_coroutine();
    }
private:
    Function<Task()> m_coroutine;
};

static void callCoroutine(auto coroutine)
{
    CoroutineCaller* caller = new CoroutineCaller();
    caller->setCoroutine([caller, coroutine = WTFMove(coroutine)] () mutable -> Task {
        co_await coroutine();
        delete caller;
    });
}

template<typename T>
class AwaitableTaskWithCompletionHandler {
public:
    using Task = CompletionHandler<void(CompletionHandler<void(T&&)>&&)>;
    AwaitableTaskWithCompletionHandler(Task&& task)
        : m_task(WTFMove(task)) { }
    bool await_ready() { return !!m_result; }
    void await_suspend(std::coroutine_handle<> handle)
    {
        m_task([this, handle] (T&& result) mutable {
            m_result = WTFMove(result);
            handle();
        });
    }
    T await_resume() { return *std::exchange(m_result, { }); }
private:
    Task m_task;
    std::optional<T> m_result;
};

}
