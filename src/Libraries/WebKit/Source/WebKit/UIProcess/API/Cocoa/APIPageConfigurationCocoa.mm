/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 20, 2022.
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
#import "APIPageConfiguration.h"

#import "WKWebViewConfigurationInternal.h"
#import <wtf/cocoa/RuntimeApplicationChecksCocoa.h>

namespace API {

#if PLATFORM(IOS_FAMILY)
bool PageConfiguration::Data::defaultShouldDecidePolicyBeforeLoadingQuickLookPreview()
{
#if USE(QUICK_LOOK)
    static bool shouldDecide = linkedOnOrAfterSDKWithBehavior(SDKAlignedBehavior::DecidesPolicyBeforeLoadingQuickLookPreview);
    return shouldDecide;
#else
    return false;
#endif
}

WebKit::DragLiftDelay PageConfiguration::Data::defaultDragLiftDelay()
{
    return fromWKDragLiftDelay(toDragLiftDelay([[NSUserDefaults standardUserDefaults] integerForKey:@"WebKitDebugDragLiftDelay"]));
}
#endif

uintptr_t PageConfiguration::Data::defaultMediaTypesRequiringUserActionForPlayback()
{
#if PLATFORM(IOS_FAMILY)
#if !PLATFORM(WATCHOS)
    if (linkedOnOrAfterSDKWithBehavior(SDKAlignedBehavior::MediaTypesRequiringUserActionForPlayback))
        return WKAudiovisualMediaTypeAudio;
#endif // !PLATFORM(WATCHOS)
    return WKAudiovisualMediaTypeAll;
#else // PLATFORM(IOS_FAMILY)
    return WKAudiovisualMediaTypeNone;
#endif // PLATFORM(IOS_FAMILY)
}

} // namespace API
