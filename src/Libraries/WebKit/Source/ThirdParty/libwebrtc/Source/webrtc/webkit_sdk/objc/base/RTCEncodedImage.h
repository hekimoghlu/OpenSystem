/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 6, 2023.
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

#import "RTCMacros.h"
#import "RTCVideoFrame.h"

NS_ASSUME_NONNULL_BEGIN

/** Represents an encoded frame's type. */
typedef NS_ENUM(NSUInteger, RTCFrameType) {
  RTCFrameTypeEmptyFrame = 0,
  RTCFrameTypeAudioFrameSpeech = 1,
  RTCFrameTypeAudioFrameCN = 2,
  RTCFrameTypeVideoFrameKey = 3,
  RTCFrameTypeVideoFrameDelta = 4,
};

typedef NS_ENUM(NSUInteger, RTCVideoContentType) {
  RTCVideoContentTypeUnspecified,
  RTCVideoContentTypeScreenshare,
};

/** Represents an encoded frame. Corresponds to webrtc::EncodedImage. */
RTC_OBJC_EXPORT
__attribute__((objc_runtime_name("WK_RTCEncodedImage")))
@interface RTCEncodedImage : NSObject

@property(nonatomic, strong) NSData *buffer;
@property(nonatomic, assign) int32_t encodedWidth;
@property(nonatomic, assign) int32_t encodedHeight;
@property(nonatomic, assign) int64_t timeStamp;
@property(nonatomic, assign) uint64_t duration;
@property(nonatomic, assign) int64_t captureTimeMs;
@property(nonatomic, assign) int64_t ntpTimeMs;
@property(nonatomic, assign) uint8_t flags;
@property(nonatomic, assign) int64_t encodeStartMs;
@property(nonatomic, assign) int64_t encodeFinishMs;
@property(nonatomic, assign) RTCFrameType frameType;
@property(nonatomic, assign) RTCVideoRotation rotation;
@property(nonatomic, assign) BOOL completeFrame;
@property(nonatomic, assign) int32_t temporalIndex;
@property(nonatomic, strong) NSNumber *qp;
@property(nonatomic, assign) RTCVideoContentType contentType;

@end

NS_ASSUME_NONNULL_END
