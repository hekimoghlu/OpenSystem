/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 27, 2024.
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
#import "WebProcessPool.h"
#import "WebProcessProxy.h"

#if PLATFORM(MAC)

#import "AuxiliaryProcess.h"
#import "CodeSigning.h"
#import "WKFullKeyboardAccessWatcher.h"
#import "WebProcessMessages.h"
#import <signal.h>
#import <wtf/ProcessPrivilege.h>

namespace WebKit {

bool WebProcessProxy::fullKeyboardAccessEnabled()
{
    return [WKFullKeyboardAccessWatcher fullKeyboardAccessEnabled];
}

bool WebProcessProxy::shouldAllowNonValidInjectedCode() const
{
    if (!AuxiliaryProcess::isSystemWebKit())
        return false;

    static bool isPlatformBinary = currentProcessIsPlatformBinary();
    if (isPlatformBinary)
        return false;

    const String& path = m_processPool->configuration().injectedBundlePath();
    return !path.isEmpty() && !path.startsWith("/System/"_s);
}

void WebProcessProxy::platformSuspendProcess()
{
    m_platformSuspendDidReleaseNearSuspendedAssertion = throttler().isHoldingNearSuspendedAssertion();
    protectedThrottler()->setShouldTakeNearSuspendedAssertion(false);
}

void WebProcessProxy::platformResumeProcess()
{
    if (m_platformSuspendDidReleaseNearSuspendedAssertion) {
        m_platformSuspendDidReleaseNearSuspendedAssertion = false;
        protectedThrottler()->setShouldTakeNearSuspendedAssertion(true);
    }
}

} // namespace WebKit

#endif // PLATFORM(MAC)
