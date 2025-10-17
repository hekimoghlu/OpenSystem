/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 3, 2022.
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
#import <AVFoundation/AVFoundation.h>
#import <Foundation/Foundation.h>

#import "RTCMacros.h"

NS_ASSUME_NONNULL_BEGIN

typedef NS_ENUM(NSInteger, RTCVideoRotation) {
  RTCVideoRotation_0 = 0,
  RTCVideoRotation_90 = 90,
  RTCVideoRotation_180 = 180,
  RTCVideoRotation_270 = 270,
};

@protocol RTCVideoFrameBuffer;

// RTCVideoFrame is an ObjectiveC version of webrtc::VideoFrame.
RTC_OBJC_EXPORT
__attribute__((objc_runtime_name("WK_RTCVideoFrame")))
@interface RTCVideoFrame : NSObject

/** Width without rotation applied. */
@property(nonatomic, readonly) int width;

/** Height without rotation applied. */
@property(nonatomic, readonly) int height;
@property(nonatomic, readonly) RTCVideoRotation rotation;

/** Timestamp in nanoseconds. */
@property(nonatomic, readonly) int64_t timeStampNs;

/** Timestamp 90 kHz. */
@property(nonatomic, assign) int64_t timeStamp;

@property(nonatomic, assign) uint64_t duration;

@property(nonatomic, readonly) id<RTCVideoFrameBuffer> buffer;

- (instancetype)init NS_UNAVAILABLE;
- (instancetype) new NS_UNAVAILABLE;

/** Initialize an RTCVideoFrame from a pixel buffer, rotation, and timestamp.
 *  Deprecated - initialize with a RTCCVPixelBuffer instead
 */
- (instancetype)initWithPixelBuffer:(CVPixelBufferRef)pixelBuffer
                           rotation:(RTCVideoRotation)rotation
                        timeStampNs:(int64_t)timeStampNs
    DEPRECATED_MSG_ATTRIBUTE("use initWithBuffer instead");

/** Initialize an RTCVideoFrame from a pixel buffer combined with cropping and
 *  scaling. Cropping will be applied first on the pixel buffer, followed by
 *  scaling to the final resolution of scaledWidth x scaledHeight.
 */
- (instancetype)initWithPixelBuffer:(CVPixelBufferRef)pixelBuffer
                        scaledWidth:(int)scaledWidth
                       scaledHeight:(int)scaledHeight
                          cropWidth:(int)cropWidth
                         cropHeight:(int)cropHeight
                              cropX:(int)cropX
                              cropY:(int)cropY
                           rotation:(RTCVideoRotation)rotation
                        timeStampNs:(int64_t)timeStampNs
    DEPRECATED_MSG_ATTRIBUTE("use initWithBuffer instead");

/** Initialize an RTCVideoFrame from a frame buffer, rotation, and timestamp.
 */
- (instancetype)initWithBuffer:(id<RTCVideoFrameBuffer>)frameBuffer
                      rotation:(RTCVideoRotation)rotation
                   timeStampNs:(int64_t)timeStampNs;

/** Return a frame that is guaranteed to be I420, i.e. it is possible to access
 *  the YUV data on it.
 */
- (RTCVideoFrame *)newI420VideoFrame;

@end

NS_ASSUME_NONNULL_END
