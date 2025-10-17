/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 20, 2022.
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
#import "RTCMediaSource.h"

#include "api/media_stream_interface.h"

NS_ASSUME_NONNULL_BEGIN

@class RTCPeerConnectionFactory;

typedef NS_ENUM(NSInteger, RTCMediaSourceType) {
  RTCMediaSourceTypeAudio,
  RTCMediaSourceTypeVideo,
};

@interface RTCMediaSource ()

@property(nonatomic, readonly) rtc::scoped_refptr<webrtc::MediaSourceInterface> nativeMediaSource;

- (instancetype)initWithFactory:(RTCPeerConnectionFactory *)factory
              nativeMediaSource:(rtc::scoped_refptr<webrtc::MediaSourceInterface>)nativeMediaSource
                           type:(RTCMediaSourceType)type NS_DESIGNATED_INITIALIZER;

+ (webrtc::MediaSourceInterface::SourceState)nativeSourceStateForState:(RTCSourceState)state;

+ (RTCSourceState)sourceStateForNativeState:(webrtc::MediaSourceInterface::SourceState)nativeState;

+ (NSString *)stringForState:(RTCSourceState)state;

@end

NS_ASSUME_NONNULL_END
