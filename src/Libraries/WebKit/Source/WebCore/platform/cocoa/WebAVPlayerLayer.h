/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 31, 2023.
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

#if HAVE(AVKIT)

#include "FloatRect.h"
#include <CoreGraphics/CGGeometry.h>
#include <QuartzCore/CALayer.h>
#include <pal/spi/cocoa/FoundationSPI.h>

OBJC_CLASS AVPlayerController;
OBJC_CLASS NSDictionary;
OBJC_CLASS NSString;

namespace WebCore {
class VideoPresentationModel;
}

WEBCORE_EXPORT @interface WebAVPlayerLayer : CALayer
@property (nonatomic, retain, nullable) NSString *videoGravity;
@property (nonatomic, getter=isReadyForDisplay) BOOL readyForDisplay;
@property (nonatomic, assign, nullable) WebCore::VideoPresentationModel* presentationModel;
@property (nonatomic, retain, nonnull) AVPlayerController *playerController;
@property (nonatomic, retain, nonnull) CALayer *videoSublayer;
@property (nonatomic, retain, nullable) CALayer *captionsLayer;
@property (nonatomic, copy, nullable) NSDictionary *pixelBufferAttributes;
@property CGSize videoDimensions;
@property (nonatomic) NSEdgeInsets legibleContentInsets;
- (WebCore::FloatRect)calculateTargetVideoFrame;
@end

#endif // HAVE(AVKIT)
