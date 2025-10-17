/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 10, 2023.
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
#if PLATFORM(IOS_FAMILY) && ENABLE(AIRPLAY_PICKER)

#import <Foundation/Foundation.h>

namespace WebCore {
enum class RouteSharingPolicy : uint8_t;
}

@class UIView;

@interface WKAirPlayRoutePicker : NSObject
- (void)showFromView:(UIView *)view routeSharingPolicy:(WebCore::RouteSharingPolicy)policy routingContextUID:(NSString *)contextUID hasVideo:(BOOL)hasVideo;
@end

#endif // PLATFORM(IOS_FAMILY) && ENABLE(AIRPLAY_PICKER)

