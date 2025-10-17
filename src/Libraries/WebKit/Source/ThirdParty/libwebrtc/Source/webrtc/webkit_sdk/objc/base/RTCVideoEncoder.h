/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 28, 2022.
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
#import <Foundation/Foundation.h>

#import "RTCCodecSpecificInfo.h"
#import "RTCEncodedImage.h"
#import "RTCMacros.h"
#import "RTCRtpFragmentationHeader.h"
#import "RTCVideoEncoderQpThresholds.h"
#import "RTCVideoEncoderSettings.h"
#import "RTCVideoFrame.h"

NS_ASSUME_NONNULL_BEGIN

/** Callback block for encoder. */
typedef BOOL (^RTCVideoEncoderCallback)(RTCEncodedImage *frame,
                                        id<RTCCodecSpecificInfo> info,
                                        RTCRtpFragmentationHeader* __nullable header);

typedef void (^RTCVideoEncoderDescriptionCallback)(const uint8_t* __nullable frame, size_t size);
typedef void (^RTCVideoEncoderErrorCallback)(OSStatus error);

/** Protocol for encoder implementations. */
RTC_OBJC_EXPORT
@protocol RTCVideoEncoder <NSObject>

- (void)setCallback:(RTCVideoEncoderCallback)callback;
- (NSInteger)startEncodeWithSettings:(RTCVideoEncoderSettings *)settings
                       numberOfCores:(int)numberOfCores;
- (NSInteger)releaseEncoder;
- (NSInteger)encode:(RTCVideoFrame *)frame
    codecSpecificInfo:(nullable id<RTCCodecSpecificInfo>)info
           frameTypes:(NSArray<NSNumber *> *)frameTypes;
- (int)setBitrate:(uint32_t)bitrateKbit framerate:(uint32_t)framerate;
- (NSString *)implementationName;

/** Returns QP scaling settings for encoder. The quality scaler adjusts the resolution in order to
 *  keep the QP from the encoded images within the given range. Returning nil from this function
 *  disables quality scaling. */
- (nullable RTCVideoEncoderQpThresholds *)scalingSettings;

@end

NS_ASSUME_NONNULL_END
