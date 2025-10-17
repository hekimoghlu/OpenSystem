/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 17, 2024.
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
#import "PageDebugger.h"

#if PLATFORM(MAC)

#import <wtf/ProcessPrivilege.h>

namespace WebCore {

bool PageDebugger::platformShouldContinueRunningEventLoopWhilePaused()
{
    // Be very careful before removing this code. It has been tried multiple times, always ending up
    // breaking inspection of WebKitLegacy (in MiniBrowser, in 3rd party apps, or sometimes both).
    //  - <https://webkit.org/b/117596> <rdar://problem/14133001>
    //  - <https://webkit.org/b/210177> <rdar://problem/61485723>

#if ENABLE(WEBPROCESS_NSRUNLOOP)
    if (![NSApp isRunning]) {
        [[NSRunLoop currentRunLoop] runUntilDate:[NSDate dateWithTimeIntervalSinceNow:0.05]];
        return true;
    }

    ASSERT(hasProcessPrivilege(ProcessPrivilege::CanCommunicateWithWindowServer));
#endif

    [NSApp setWindowsNeedUpdate:YES];

    if (NSEvent *event = [NSApp nextEventMatchingMask:NSEventMaskAny untilDate:[NSDate dateWithTimeIntervalSinceNow:0.05] inMode:NSDefaultRunLoopMode dequeue:YES])
        [NSApp sendEvent:event];

    return true;
}

} // namespace WebCore

#endif // PLATFORM(MAC)
