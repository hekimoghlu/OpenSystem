/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 16, 2023.
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
#include "VideoFrame.h"

#if ENABLE(VIDEO) && PLATFORM(COCOA)

#import "PixelBufferConformerCV.h"
#include "VideoFrameCV.h"
#import <JavaScriptCore/TypedArrayInlines.h>

#import <pal/cf/CoreMediaSoftLink.h>
#import "CoreVideoSoftLink.h"

namespace WebCore {

#if USE(AVFOUNDATION)
RefPtr<VideoFrameCV> VideoFrame::asVideoFrameCV()
{
    if (auto* videoFrameCV = dynamicDowncast<VideoFrameCV>(*this))
        return videoFrameCV;

    auto buffer = pixelBuffer();
    if (!buffer)
        return nullptr;
    return VideoFrameCV::create(presentationTime(), isMirrored(), rotation(), buffer);
}
#endif // USE(AVFOUNDATION)

}

#endif // ENABLE(VIDEO) && PLATFORM(COCOA)
