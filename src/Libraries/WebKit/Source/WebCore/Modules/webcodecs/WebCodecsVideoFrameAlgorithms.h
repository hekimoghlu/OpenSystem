/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 24, 2021.
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

#include "WebCodecsVideoFrame.h"

#if ENABLE(WEB_CODECS)

namespace WebCore {

bool isValidVideoFrameBufferInit(const WebCodecsVideoFrame::BufferInit&);
bool verifyRectOffsetAlignment(VideoPixelFormat, const DOMRectInit&);
bool verifyRectSizeAlignment(VideoPixelFormat, const DOMRectInit&);
ExceptionOr<DOMRectInit> parseVisibleRect(const DOMRectInit&, const std::optional<DOMRectInit>&, size_t codedWidth, size_t codedHeight, VideoPixelFormat);
size_t videoPixelFormatToPlaneCount(VideoPixelFormat);
size_t videoPixelFormatToSampleByteSizePerPlane();
size_t videoPixelFormatToSubSampling(VideoPixelFormat, size_t);

struct CombinedPlaneLayout {
    size_t allocationSize { 0 };
    Vector<ComputedPlaneLayout> computedLayouts;
};

ExceptionOr<CombinedPlaneLayout> computeLayoutAndAllocationSize(const DOMRectInit&, const std::optional<Vector<PlaneLayout>>&, VideoPixelFormat);

ExceptionOr<CombinedPlaneLayout> parseVideoFrameCopyToOptions(const WebCodecsVideoFrame&, const WebCodecsVideoFrame::CopyToOptions&);

void initializeVisibleRectAndDisplaySize(WebCodecsVideoFrame&, const WebCodecsVideoFrame::Init&, const DOMRectInit&, size_t defaultDisplayWidth, size_t defaultDisplayHeight);

VideoColorSpaceInit videoFramePickColorSpace(const std::optional<VideoColorSpaceInit>&, VideoPixelFormat);

bool validateVideoFrameInit(const WebCodecsVideoFrame::Init&, size_t codedWidth, size_t codedHeight, VideoPixelFormat);

}

#endif
