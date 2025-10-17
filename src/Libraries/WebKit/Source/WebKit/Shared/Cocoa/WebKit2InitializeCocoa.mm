/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 15, 2024.
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
#import "WebKit2Initialize.h"

#import <JavaScriptCore/InitializeThreading.h>
#import <WebCore/CommonAtomStrings.h>
#import <WebCore/WebCoreJITOperations.h>
#import <mutex>
#import <wtf/MainThread.h>
#import <wtf/RefCounted.h>
#import <wtf/WorkQueue.h>
#import <wtf/cocoa/RuntimeApplicationChecksCocoa.h>

#if PLATFORM(IOS_FAMILY)
#import <WebCore/WebCoreThreadSystemInterface.h>
#endif

#if ENABLE(LLVM_PROFILE_GENERATION)
extern "C" char __llvm_profile_filename[] = "/private/tmp/WebKitPGO/WebKit_%m_pid%p%c.profraw";
#endif

namespace WebKit {

static std::once_flag flag;

enum class WebKitProfileTag { };

static void runInitializationCode(void* = nullptr)
{
    RELEASE_ASSERT_WITH_MESSAGE([NSThread isMainThread], "InitializeWebKit2 should be called on the main thread");

    WTF::initializeMainThread();
    JSC::initialize();
    WebCore::initializeCommonAtomStrings();
#if PLATFORM(IOS_FAMILY)
    InitWebCoreThreadSystemInterface();
#endif

    WTF::RefCountedBase::enableThreadingChecksGlobally();

    WebCore::populateJITOperations();
}

void InitializeWebKit2()
{
    // Make sure the initialization code is run only once and on the main thread since things like initializeMainThread()
    // are only safe to call on the main thread.
    std::call_once(flag, [] {
        if ([NSThread isMainThread] || linkedOnOrAfterSDKWithBehavior(SDKAlignedBehavior::InitializeWebKit2MainThreadAssertion))
            runInitializationCode();
        else
            WorkQueue::main().dispatchSync([] { runInitializationCode(); });
    });
}

}
