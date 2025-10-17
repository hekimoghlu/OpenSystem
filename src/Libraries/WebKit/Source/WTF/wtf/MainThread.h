/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 20, 2024.
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

#include <stdint.h>
#include <wtf/Forward.h>
#include <wtf/Function.h>
#include <wtf/ThreadAssertions.h>
#include <wtf/ThreadingPrimitives.h>

namespace WTF {

class PrintStream;
class Thread;

// Must be called from the main thread.
WTF_EXPORT_PRIVATE void initializeMainThread();

WTF_EXPORT_PRIVATE void callOnMainThread(Function<void()>&&);
WTF_EXPORT_PRIVATE void callOnMainThreadAndWait(Function<void()>&&);
WTF_EXPORT_PRIVATE void ensureOnMainThread(Function<void()>&&); // Sync if called on main thread, async otherwise.

#if PLATFORM(COCOA)
WTF_EXPORT_PRIVATE void dispatchAsyncOnMainThreadWithWebThreadLockIfNeeded(void (^block)());
WTF_EXPORT_PRIVATE void callOnWebThreadOrDispatchAsyncOnMainThread(void (^block)());
#endif

WTF_EXPORT_PRIVATE bool isMainThread();

WTF_EXPORT_PRIVATE bool canCurrentThreadAccessThreadLocalData(Thread&);

WTF_EXPORT_PRIVATE bool isMainRunLoop();
WTF_EXPORT_PRIVATE void callOnMainRunLoop(Function<void()>&&);
WTF_EXPORT_PRIVATE void callOnMainRunLoopAndWait(Function<void()>&&);
WTF_EXPORT_PRIVATE void ensureOnMainRunLoop(Function<void()>&&); // Sync if called on main run loop, async otherwise.

#if USE(WEB_THREAD)
WTF_EXPORT_PRIVATE bool isWebThread();
WTF_EXPORT_PRIVATE bool isUIThread();
WTF_EXPORT_PRIVATE void initializeWebThread();
WTF_EXPORT_PRIVATE void initializeApplicationUIThread();
#else
inline bool isWebThread() { return isMainThread(); }
inline bool isUIThread() { return isMainThread(); }
#endif // USE(WEB_THREAD)

WTF_EXPORT_PRIVATE bool isMainThreadOrGCThread();

// NOTE: these functions are internal to the callOnMainThread implementation.
void initializeMainThreadPlatform();

// To be used with WTF_REQUIRES_CAPABILITY(mainThread). Symbol is undefined.
extern NamedAssertion& mainThread;
inline void assertIsMainThread() WTF_ASSERTS_ACQUIRED_CAPABILITY(mainThread) { ASSERT(isMainThread()); }

// To be used with WTF_REQUIRES_CAPABILITY(mainRunLoop). Symbol is undefined.
extern NamedAssertion& mainRunLoop;
inline void assertIsMainRunLoop() WTF_ASSERTS_ACQUIRED_CAPABILITY(mainRunLoop) { ASSERT(isMainRunLoop()); }

enum class DestructionThread : uint8_t { Any, Main, MainRunLoop };

} // namespace WTF

using WTF::assertIsMainRunLoop;
using WTF::assertIsMainThread;
using WTF::callOnMainRunLoop;
using WTF::callOnMainRunLoopAndWait;
using WTF::callOnMainThread;
using WTF::callOnMainThreadAndWait;
using WTF::canCurrentThreadAccessThreadLocalData;
using WTF::ensureOnMainRunLoop;
using WTF::ensureOnMainThread;
using WTF::isMainRunLoop;
using WTF::isMainThread;
using WTF::isMainThreadOrGCThread;
using WTF::isUIThread;
using WTF::isWebThread;
using WTF::mainRunLoop;
using WTF::mainThread;

#if PLATFORM(COCOA)
using WTF::dispatchAsyncOnMainThreadWithWebThreadLockIfNeeded;
using WTF::callOnWebThreadOrDispatchAsyncOnMainThread;
#endif

#if USE(WEB_THREAD)
using WTF::initializeWebThread;
using WTF::initializeApplicationUIThread;
#endif
