/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 5, 2025.
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
#import "RTCPeerConnectionFactory.h"

#include "api/scoped_refptr.h"

namespace webrtc {

class AudioDeviceModule;
class AudioEncoderFactory;
class AudioDecoderFactory;
class VideoEncoderFactory;
class VideoDecoderFactory;
class AudioProcessing;

}  // namespace webrtc

NS_ASSUME_NONNULL_BEGIN

@interface RTCPeerConnectionFactoryBuilder : NSObject

+ (RTCPeerConnectionFactoryBuilder *)builder;

- (RTCPeerConnectionFactory *)createPeerConnectionFactory;

- (void)setVideoEncoderFactory:(std::unique_ptr<webrtc::VideoEncoderFactory>)videoEncoderFactory;

- (void)setVideoDecoderFactory:(std::unique_ptr<webrtc::VideoDecoderFactory>)videoDecoderFactory;

- (void)setAudioEncoderFactory:(rtc::scoped_refptr<webrtc::AudioEncoderFactory>)audioEncoderFactory;

- (void)setAudioDecoderFactory:(rtc::scoped_refptr<webrtc::AudioDecoderFactory>)audioDecoderFactory;

- (void)setAudioDeviceModule:(rtc::scoped_refptr<webrtc::AudioDeviceModule>)audioDeviceModule;

- (void)setAudioProcessingModule:(rtc::scoped_refptr<webrtc::AudioProcessing>)audioProcessingModule;

@end

NS_ASSUME_NONNULL_END
