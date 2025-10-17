/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 11, 2024.
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
#import "RTCPeerConnectionFactory+Native.h"
#import "RTCPeerConnectionFactoryBuilder+DefaultComponents.h"

#include "api/audio_codecs/builtin_audio_decoder_factory.h"
#include "api/audio_codecs/builtin_audio_encoder_factory.h"
#import "components/video_codec/RTCVideoDecoderFactoryH264.h"
#import "components/video_codec/RTCVideoEncoderFactoryH264.h"
#include "sdk/objc/native/api/video_decoder_factory.h"
#include "sdk/objc/native/api/video_encoder_factory.h"

#if defined(WEBRTC_IOS)
#import "sdk/objc/native/api/audio_device_module.h"
#endif

@implementation RTCPeerConnectionFactoryBuilder (DefaultComponents)

+ (RTCPeerConnectionFactoryBuilder *)defaultBuilder {
  RTCPeerConnectionFactoryBuilder *builder = [[RTCPeerConnectionFactoryBuilder alloc] init];
  auto audioEncoderFactory = webrtc::CreateBuiltinAudioEncoderFactory();
  [builder setAudioEncoderFactory:audioEncoderFactory];

  auto audioDecoderFactory = webrtc::CreateBuiltinAudioDecoderFactory();
  [builder setAudioDecoderFactory:audioDecoderFactory];

  auto videoEncoderFactory =
      webrtc::ObjCToNativeVideoEncoderFactory([[RTCVideoEncoderFactoryH264 alloc] init]);
  [builder setVideoEncoderFactory:std::move(videoEncoderFactory)];

  auto videoDecoderFactory =
      webrtc::ObjCToNativeVideoDecoderFactory([[RTCVideoDecoderFactoryH264 alloc] init]);
  [builder setVideoDecoderFactory:std::move(videoDecoderFactory)];

#if defined(WEBRTC_IOS)
  [builder setAudioDeviceModule:webrtc::CreateAudioDeviceModule()];
#endif
  return builder;
}

@end
