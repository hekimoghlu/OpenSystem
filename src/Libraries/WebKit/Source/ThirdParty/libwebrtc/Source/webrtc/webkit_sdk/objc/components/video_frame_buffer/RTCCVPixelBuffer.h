/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 5, 2022.
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

#import "RTCMacros.h"
#import "RTCVideoFrameBuffer.h"

NS_ASSUME_NONNULL_BEGIN

/** RTCVideoFrameBuffer containing a CVPixelBufferRef */
RTC_OBJC_EXPORT
__attribute__((objc_runtime_name("WK_RTCCVPixelBuffer")))
@interface RTCCVPixelBuffer : NSObject <RTCVideoFrameBuffer>

@property(nonatomic, readonly) CVPixelBufferRef pixelBuffer;
@property(nonatomic, readonly) int cropX;
@property(nonatomic, readonly) int cropY;
@property(nonatomic, readonly) int cropWidth;
@property(nonatomic, readonly) int cropHeight;

+ (NSSet<NSNumber *> *)supportedPixelFormats;

- (instancetype)initWithPixelBuffer:(CVPixelBufferRef)pixelBuffer;
- (instancetype)initWithPixelBuffer:(CVPixelBufferRef)pixelBuffer
                       adaptedWidth:(int)adaptedWidth
                      adaptedHeight:(int)adaptedHeight
                          cropWidth:(int)cropWidth
                         cropHeight:(int)cropHeight
                              cropX:(int)cropX
                              cropY:(int)cropY;

- (BOOL)requiresCropping;
- (BOOL)requiresScalingToWidth:(int)width height:(int)height;
- (int)bufferSizeForCroppingAndScalingToWidth:(int)width height:(int)height;

/** The minimum size of the |tmpBuffer| must be the number of bytes returned from the
 * bufferSizeForCroppingAndScalingToWidth:height: method.
 * If that size is 0, the |tmpBuffer| may be nil.
 */
- (BOOL)cropAndScaleTo:(CVPixelBufferRef)outputPixelBuffer
        withTempBuffer:(nullable uint8_t *)tmpBuffer;

#if defined(WEBRTC_WEBKIT_BUILD)
- (void)close;
#endif
@end

NS_ASSUME_NONNULL_END
