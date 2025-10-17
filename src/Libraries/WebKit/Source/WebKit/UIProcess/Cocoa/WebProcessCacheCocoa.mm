/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 27, 2025.
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
#import "WebProcessCache.h"

namespace WebKit {

void WebProcessCache::platformInitialize()
{
    Seconds cachedProcessLifetimeOverride([[NSUserDefaults standardUserDefaults] doubleForKey:@"WebProcessCacheCachedProcessLifetimeInSeconds"]);
    if (cachedProcessLifetimeOverride > 0_s && cachedProcessLifetimeOverride <= 24_h) {
        cachedProcessLifetime = cachedProcessLifetimeOverride;
        WTFLogAlways("Warning: WebProcessCache cachedProcessLifetime was overriden via user defaults and is now %g seconds", cachedProcessLifetimeOverride.seconds());
    }

    Seconds clearingDelayAfterApplicationResignsActiveOverride([[NSUserDefaults standardUserDefaults] doubleForKey:@"WebProcessCacheClearingDelayAfterApplicationResignsActiveInSeconds"]);
    if (clearingDelayAfterApplicationResignsActiveOverride > 0_s && clearingDelayAfterApplicationResignsActiveOverride <= 1_h) {
        clearingDelayAfterApplicationResignsActive = clearingDelayAfterApplicationResignsActiveOverride;
        WTFLogAlways("Warning: WebProcessCache clearingDelayAfterApplicationResignsActive was overriden via user defaults and is now %g seconds", clearingDelayAfterApplicationResignsActiveOverride.seconds());
    }
}

} // namespace WebKit
