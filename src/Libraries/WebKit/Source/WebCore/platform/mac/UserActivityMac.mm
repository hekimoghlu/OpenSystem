/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 5, 2024.
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
#import "UserActivity.h"

#import "Logging.h"

namespace WebCore {

#if HAVE(NS_ACTIVITY)

UserActivity::Impl::Impl(ASCIILiteral descriptionLiteral)
    : m_description(descriptionLiteral.createNSString())
{
}

void UserActivity::Impl::beginActivity()
{
    if (!m_activity) {
        RELEASE_LOG(ActivityState, "%p - UserActivity::Impl::beginActivity: description=%" PUBLIC_LOG_STRING, this, [m_description UTF8String]);
        NSActivityOptions options = (NSActivityUserInitiatedAllowingIdleSystemSleep | NSActivityLatencyCritical) & ~(NSActivitySuddenTerminationDisabled | NSActivityAutomaticTerminationDisabled);
        m_activity = [[NSProcessInfo processInfo] beginActivityWithOptions:options reason:m_description.get()];
    }
}

void UserActivity::Impl::endActivity()
{
    RELEASE_LOG(ActivityState, "%p - UserActivity::Impl::endActivity: description=%" PUBLIC_LOG_STRING, this, [m_description UTF8String]);
    [[NSProcessInfo processInfo] endActivity:m_activity.get()];
    m_activity.clear();
}

#endif

} // namespace WebCore
