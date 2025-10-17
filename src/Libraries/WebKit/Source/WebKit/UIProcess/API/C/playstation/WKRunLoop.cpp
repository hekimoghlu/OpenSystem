/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 12, 2023.
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
#include "WKRunLoop.h"

#include "WebKit2Initialize.h"
#include <wtf/MainThread.h>
#include <wtf/RunLoop.h>

void WKRunLoopRunMain()
{
    RunLoop::main().run();
}

void WKRunLoopStopMain()
{
    RunLoop::main().stop();
}

void WKRunLoopCallOnMainThread(WKRunLoopCallback callback, void* userData)
{
    if (!callback)
        return;

    callOnMainThread([callback, userData]() {
        callback(userData);
    });
}

void WKRunLoopInitializeMain()
{
    // FIXME:
    // Initializing RunLoop by InitializeWebKit2().
    // WKRunLoopInitializeMain() should be renamed to more straightforward API name to wrap InitializeWebKit2().
    WebKit::InitializeWebKit2();
}
