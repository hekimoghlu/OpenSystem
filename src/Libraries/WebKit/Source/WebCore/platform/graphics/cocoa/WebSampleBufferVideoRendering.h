/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 6, 2022.
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
#import <pal/spi/cocoa/AVFoundationSPI.h>

NS_ASSUME_NONNULL_BEGIN

@protocol WebSampleBufferVideoRendering <AVQueuedSampleBufferRendering>
- (void)expectMinimumUpcomingSampleBufferPresentationTime:(CMTime)minimumUpcomingPresentationTime;
- (void)resetUpcomingSampleBufferPresentationTimeExpectations;
- (nullable CVPixelBufferRef)copyDisplayedPixelBuffer;
- (void)prerollDecodeWithCompletionHandler:(void (^)(BOOL success))block;
- (nullable AVVideoPerformanceMetrics *)videoPerformanceMetrics;
@property BOOL preventsAutomaticBackgroundingDuringVideoPlayback;
@property BOOL preventsDisplaySleepDuringVideoPlayback;
@property (readonly) BOOL requiresFlushToResumeDecoding;
@property (readonly) AVQueuedSampleBufferRenderingStatus status;
@property (readonly, nullable) NSError *error;
@end

@interface AVSampleBufferDisplayLayer (WebCoreExtras) <WebSampleBufferVideoRendering>
@end

#if HAVE(AVSAMPLEBUFFERVIDEORENDERER)
@interface AVSampleBufferVideoRenderer (WebCoreExtras) <WebSampleBufferVideoRendering>
@end
#endif

NS_ASSUME_NONNULL_END
