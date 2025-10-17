/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 14, 2021.
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
#pragma once

#if HAVE(CORE_ANIMATION_FRAME_RATE_RANGE)

#import <pal/spi/cocoa/QuartzCoreSPI.h>

namespace WebKit {

#define WEBKIT_HIGH_FRAME_RATE_REASON_COMPONENT 44

static CAFrameRateRange highFrameRateRange()
{
    static CAFrameRateRange highFrameRateRange = CAFrameRateRangeMake(80, 120, 120);
    return highFrameRateRange;
}

static CAHighFrameRateReason webAnimationHighFrameRateReason = CAHighFrameRateReasonMake(WEBKIT_HIGH_FRAME_RATE_REASON_COMPONENT, 1);
static CAHighFrameRateReason keyboardScrollingAnimationHighFrameRateReason = CAHighFrameRateReasonMake(WEBKIT_HIGH_FRAME_RATE_REASON_COMPONENT, 2);
static CAHighFrameRateReason preferPageRenderingUpdatesNear60FPSDisabledHighFrameRateReason = CAHighFrameRateReasonMake(WEBKIT_HIGH_FRAME_RATE_REASON_COMPONENT, 3);

}

#endif // HAVE(CORE_ANIMATION_FRAME_RATE_RANGE)
