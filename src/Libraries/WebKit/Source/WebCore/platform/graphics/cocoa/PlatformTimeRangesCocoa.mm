/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 26, 2024.
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
#import "PlatformTimeRanges.h"

#if PLATFORM(COCOA)

#import <AVFoundation/AVFoundation.h>
#import <pal/avfoundation/MediaTimeAVFoundation.h>

#import <pal/cf/CoreMediaSoftLink.h>

namespace WebCore {

RetainPtr<NSArray> makeNSArray(const PlatformTimeRanges& timeRanges)
{
    RetainPtr ranges = adoptNS([[NSMutableArray alloc] initWithCapacity:timeRanges.length()]);

    for (unsigned i = 0; i < timeRanges.length(); ++i) {
        bool startValid;
        MediaTime start = timeRanges.start(i, startValid);
        RELEASE_ASSERT(startValid);

        bool endValid;
        MediaTime end = timeRanges.end(i, endValid);
        RELEASE_ASSERT(endValid);

        [ranges addObject:[NSValue valueWithCMTimeRange:PAL::CMTimeRangeMake(PAL::toCMTime(start), PAL::toCMTime(end - start))]];
    }

    return adoptNS([ranges copy]);
}

} // namespace WebCore

#endif // PLATFORM(COCOA)
