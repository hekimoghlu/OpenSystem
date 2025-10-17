/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 17, 2022.
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
#import "LockdownModeCocoa.h"

#if HAVE(LOCKDOWN_MODE_FRAMEWORK)

#import <LockdownMode/LockdownMode.h>
#import <sys/sysctl.h>
#import <wtf/NeverDestroyed.h>
#import <wtf/SoftLinking.h>

OBJC_CLASS LockdownModeManager;

SOFT_LINK_PRIVATE_FRAMEWORK_OPTIONAL(LockdownMode)
SOFT_LINK_CLASS_OPTIONAL(LockdownMode, LockdownModeManager)

namespace PAL {

bool isLockdownModeEnabled()
{
    if (LockdownModeLibrary())
        return [(LockdownModeManager *)[getLockdownModeManagerClass() shared] enabled];

    // FIXME(<rdar://108208100>): Remove this fallback once recoveryOS includes the framework.
    uint64_t ldmState = 0;
    size_t sysCtlLen = sizeof(ldmState);
    if (!sysctlbyname("security.mac.lockdown_mode_state", &ldmState, &sysCtlLen, NULL, 0))
        return ldmState == 1;

    return false;
}

static std::optional<bool>& isLockdownModeEnabledForCurrentProcessCached()
{
    static NeverDestroyed<std::optional<bool>> cachedIsLockdownModeEnabledForCurrentProcess;
    return cachedIsLockdownModeEnabledForCurrentProcess;
}

bool isLockdownModeEnabledForCurrentProcess()
{
    return isLockdownModeEnabledForCurrentProcessCached().value_or(isLockdownModeEnabled());
}

void setLockdownModeEnabledForCurrentProcess(bool isLockdownModeEnabled)
{
    isLockdownModeEnabledForCurrentProcessCached() = isLockdownModeEnabled;
}

} // namespace PAL

#endif
