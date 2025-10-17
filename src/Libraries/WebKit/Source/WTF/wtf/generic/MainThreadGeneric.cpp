/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 3, 2024.
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
#include <pthread.h>

#include <wtf/RunLoop.h>

namespace WTF {

static pthread_t s_mainThread;

void initializeMainThreadPlatform()
{
    // WebKit APIs must be consistently used from exactly one thread. The thread that initializes WebKit
    // is considered the "WebKit main thread," and it is an error to use WebKit APIs from any other thread.
    // The WebKit main thread need not be the application's actual OS-level main thread, which might be
    // controlled by a language runtime or virtual machine; for example, in Eclipse, the OS main thread is
    // controlled by the JVM, while the separate WebKit main thread runs all the GUI and WebKit stuff.
    s_mainThread = pthread_self();
}

bool isMainThread()
{
    return pthread_equal(pthread_self(), s_mainThread);
}

} // namespace WTF
