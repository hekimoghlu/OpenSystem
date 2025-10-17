/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 30, 2021.
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
#import "config.h"
#import <wtf/MainThread.h>

#import <CoreFoundation/CoreFoundation.h>
#import <Foundation/NSThread.h>
#import <dispatch/dispatch.h>
#import <stdio.h>
#import <wtf/Assertions.h>
#import <wtf/BlockPtr.h>
#import <wtf/Logging.h>
#import <wtf/RetainPtr.h>
#import <wtf/RunLoop.h>
#import <wtf/SchedulePair.h>
#import <wtf/Threading.h>

#if USE(WEB_THREAD)
#import <wtf/ios/WebCoreThread.h>
#endif

namespace WTF {

#if USE(WEB_THREAD)
// When the Web thread is enabled, we consider it to be the main thread, not pthread main.
static pthread_t s_webThreadPthread;

static Thread* s_applicationUIThread;
static Thread* s_webThread;
#endif

void initializeMainThreadPlatform()
{
    if (!pthread_main_np())
        RELEASE_LOG_FAULT(Threading, "WebKit Threading Violation - initial use of WebKit from a secondary thread.");
    ASSERT(pthread_main_np());
}

void dispatchAsyncOnMainThreadWithWebThreadLockIfNeeded(void (^block)())
{
#if USE(WEB_THREAD)
    if (WebCoreWebThreadIsEnabled && WebCoreWebThreadIsEnabled()) {
        RunLoop::main().dispatch([block = makeBlockPtr(block)] {
            WebCoreWebThreadLock();
            block();
        });
        return;
    }
#endif
    RunLoop::main().dispatch([block = makeBlockPtr(block)] {
        block();
    });
}

void callOnWebThreadOrDispatchAsyncOnMainThread(void (^block)())
{
#if USE(WEB_THREAD)
    if (WebCoreWebThreadIsEnabled && WebCoreWebThreadIsEnabled()) {
        WebCoreWebThreadRun(block);
        return;
    }
#endif
    RunLoop::main().dispatch([block = makeBlockPtr(block)] {
        block();
    });
}

#if USE(WEB_THREAD)

static bool webThreadIsUninitializedOrLockedOrDisabled()
{
    return !WebCoreWebThreadIsLockedOrDisabled || WebCoreWebThreadIsLockedOrDisabled();
}

bool isMainThread()
{
    return (isWebThread() || pthread_main_np()) && webThreadIsUninitializedOrLockedOrDisabled();
}

bool isUIThread()
{
    return pthread_main_np();
}

// Keep in mind that isWebThread can be called even when destroying the current thread.
bool isWebThread()
{
    return pthread_equal(pthread_self(), s_webThreadPthread);
}

void initializeApplicationUIThread()
{
    ASSERT(pthread_main_np());
    s_applicationUIThread = &Thread::current();
}

void initializeWebThread()
{
    static std::once_flag initializeKey;
    std::call_once(initializeKey, [] {
        ASSERT(!pthread_main_np());
        s_webThreadPthread = pthread_self();
        s_webThread = &Thread::current();
        RunLoop::initializeWeb();
    });
}

bool canCurrentThreadAccessThreadLocalData(Thread& thread)
{
    Thread& currentThread = Thread::current();
    if (&thread == &currentThread)
        return true;

    if (&thread == s_webThread || &thread == s_applicationUIThread)
        return (&currentThread == s_webThread || &currentThread == s_applicationUIThread) && webThreadIsUninitializedOrLockedOrDisabled();

    return false;
}

#else

bool isMainThread()
{
    return pthread_main_np();
}

#endif // USE(WEB_THREAD)

} // namespace WTF
