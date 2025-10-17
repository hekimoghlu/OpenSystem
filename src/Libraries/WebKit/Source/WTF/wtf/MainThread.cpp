/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 24, 2022.
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
#include "config.h"
#include <wtf/MainThread.h>

#include <mutex>
#include <wtf/Deque.h>
#include <wtf/Lock.h>
#include <wtf/MonotonicTime.h>
#include <wtf/NeverDestroyed.h>
#include <wtf/RunLoop.h>
#include <wtf/StdLibExtras.h>
#include <wtf/Threading.h>
#include <wtf/WorkQueue.h>
#include <wtf/threads/BinarySemaphore.h>

namespace WTF {

void initializeMainThread()
{
    static std::once_flag initializeKey;
    std::call_once(initializeKey, [] {
        initialize();
        initializeMainThreadPlatform();
        RunLoop::initializeMain();
    });
}

#if !USE(WEB_THREAD)
bool canCurrentThreadAccessThreadLocalData(Thread& thread)
{
    return &thread == &Thread::current();
}
#endif

bool isMainRunLoop()
{
    return RunLoop::isMain();
}

void callOnMainRunLoop(Function<void()>&& function)
{
    RunLoop::main().dispatch(WTFMove(function));
}

void ensureOnMainRunLoop(Function<void()>&& function)
{
    if (RunLoop::isMain())
        function();
    else
        RunLoop::main().dispatch(WTFMove(function));
}

void callOnMainThread(Function<void()>&& function)
{
#if USE(WEB_THREAD)
    if (auto* webRunLoop = RunLoop::webIfExists()) {
        webRunLoop->dispatch(WTFMove(function));
        return;
    }
#endif

    RunLoop::main().dispatch(WTFMove(function));
}

void ensureOnMainThread(Function<void()>&& function)
{
    if (isMainThread())
        function();
    else
        callOnMainThread(WTFMove(function));
}

bool isMainThreadOrGCThread()
{
    if (Thread::mayBeGCThread())
        return true;

    return isMainThread();
}

enum class MainStyle : bool {
    Thread,
    RunLoop
};

template <MainStyle mainStyle>
static void callOnMainAndWait(Function<void()>&& function)
{

    if (mainStyle == MainStyle::Thread ? isMainThread() : isMainRunLoop()) {
        function();
        return;
    }

    BinarySemaphore semaphore;
    auto functionImpl = [&semaphore, function = WTFMove(function)] {
        function();
        semaphore.signal();
    };

    switch (mainStyle) {
    case MainStyle::Thread:
        callOnMainThread(WTFMove(functionImpl));
        break;
    case MainStyle::RunLoop:
        callOnMainRunLoop(WTFMove(functionImpl));
    };
    semaphore.wait();
}

void callOnMainRunLoopAndWait(Function<void()>&& function)
{
    callOnMainAndWait<MainStyle::RunLoop>(WTFMove(function));
}

void callOnMainThreadAndWait(Function<void()>&& function)
{
    callOnMainAndWait<MainStyle::Thread>(WTFMove(function));
}

} // namespace WTF
