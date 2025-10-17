/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 20, 2023.
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
#import "RTCPeerConnectionFactoryBuilder.h"
#import "RTCPeerConnectionFactory+Native.h"

#include "api/audio_codecs/audio_decoder_factory.h"
#include "api/audio_codecs/audio_encoder_factory.h"
#include "api/transport/media/media_transport_interface.h"
#include "api/video_codecs/video_decoder_factory.h"
#include "api/video_codecs/video_encoder_factory.h"
#include "modules/audio_device/include/audio_device.h"
#include "modules/audio_processing/include/audio_processing.h"

@implementation RTCPeerConnectionFactoryBuilder {
  std::unique_ptr<webrtc::VideoEncoderFactory> _videoEncoderFactory;
  std::unique_ptr<webrtc::VideoDecoderFactory> _videoDecoderFactory;
  rtc::scoped_refptr<webrtc::AudioEncoderFactory> _audioEncoderFactory;
  rtc::scoped_refptr<webrtc::AudioDecoderFactory> _audioDecoderFactory;
  rtc::scoped_refptr<webrtc::AudioDeviceModule> _audioDeviceModule;
  rtc::scoped_refptr<webrtc::AudioProcessing> _audioProcessingModule;
  std::unique_ptr<webrtc::MediaTransportFactory> _mediaTransportFactory;
}

+ (RTCPeerConnectionFactoryBuilder *)builder {
  return [[RTCPeerConnectionFactoryBuilder alloc] init];
}

- (RTCPeerConnectionFactory *)createPeerConnectionFactory {
  RTCPeerConnectionFactory *factory = [RTCPeerConnectionFactory alloc];
  return [factory initWithNativeAudioEncoderFactory:_audioEncoderFactory
                          nativeAudioDecoderFactory:_audioDecoderFactory
                          nativeVideoEncoderFactory:std::move(_videoEncoderFactory)
                          nativeVideoDecoderFactory:std::move(_videoDecoderFactory)
                                  audioDeviceModule:_audioDeviceModule
                              audioProcessingModule:_audioProcessingModule
                              mediaTransportFactory:std::move(_mediaTransportFactory)];
}

- (void)setVideoEncoderFactory:(std::unique_ptr<webrtc::VideoEncoderFactory>)videoEncoderFactory {
  _videoEncoderFactory = std::move(videoEncoderFactory);
}

- (void)setVideoDecoderFactory:(std::unique_ptr<webrtc::VideoDecoderFactory>)videoDecoderFactory {
  _videoDecoderFactory = std::move(videoDecoderFactory);
}

- (void)setAudioEncoderFactory:
        (rtc::scoped_refptr<webrtc::AudioEncoderFactory>)audioEncoderFactory {
  _audioEncoderFactory = audioEncoderFactory;
}

- (void)setAudioDecoderFactory:
        (rtc::scoped_refptr<webrtc::AudioDecoderFactory>)audioDecoderFactory {
  _audioDecoderFactory = audioDecoderFactory;
}

- (void)setAudioDeviceModule:(rtc::scoped_refptr<webrtc::AudioDeviceModule>)audioDeviceModule {
  _audioDeviceModule = audioDeviceModule;
}

- (void)setAudioProcessingModule:
        (rtc::scoped_refptr<webrtc::AudioProcessing>)audioProcessingModule {
  _audioProcessingModule = audioProcessingModule;
}

@end
