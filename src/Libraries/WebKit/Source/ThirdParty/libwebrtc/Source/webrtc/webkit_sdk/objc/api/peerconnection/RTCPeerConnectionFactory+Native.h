/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 22, 2022.
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
class MediaTransportFactory;
class NetworkControllerFactoryInterface;
class VideoEncoderFactory;
class VideoDecoderFactory;
class AudioProcessing;
struct PeerConnectionDependencies;

}  // namespace webrtc

NS_ASSUME_NONNULL_BEGIN

/**
 * This class extension exposes methods that work directly with injectable C++ components.
 */
@interface RTCPeerConnectionFactory ()

- (instancetype)initNative NS_DESIGNATED_INITIALIZER;

/* Initializer used when WebRTC is compiled with no media support */
- (instancetype)initWithNoMedia;

/* Initialize object with injectable native audio/video encoder/decoder factories */
- (instancetype)initWithNativeAudioEncoderFactory:
                    (rtc::scoped_refptr<webrtc::AudioEncoderFactory>)audioEncoderFactory
                        nativeAudioDecoderFactory:
                            (rtc::scoped_refptr<webrtc::AudioDecoderFactory>)audioDecoderFactory
                        nativeVideoEncoderFactory:
                            (std::unique_ptr<webrtc::VideoEncoderFactory>)videoEncoderFactory
                        nativeVideoDecoderFactory:
                            (std::unique_ptr<webrtc::VideoDecoderFactory>)videoDecoderFactory
                                audioDeviceModule:
                                    (nullable webrtc::AudioDeviceModule *)audioDeviceModule
                            audioProcessingModule:
                                (rtc::scoped_refptr<webrtc::AudioProcessing>)audioProcessingModule;

- (instancetype)
    initWithNativeAudioEncoderFactory:
        (rtc::scoped_refptr<webrtc::AudioEncoderFactory>)audioEncoderFactory
            nativeAudioDecoderFactory:
                (rtc::scoped_refptr<webrtc::AudioDecoderFactory>)audioDecoderFactory
            nativeVideoEncoderFactory:
                (std::unique_ptr<webrtc::VideoEncoderFactory>)videoEncoderFactory
            nativeVideoDecoderFactory:
                (std::unique_ptr<webrtc::VideoDecoderFactory>)videoDecoderFactory
                    audioDeviceModule:(nullable webrtc::AudioDeviceModule *)audioDeviceModule
                audioProcessingModule:
                    (rtc::scoped_refptr<webrtc::AudioProcessing>)audioProcessingModule
                mediaTransportFactory:
                    (std::unique_ptr<webrtc::MediaTransportFactory>)mediaTransportFactory;

- (instancetype)
    initWithNativeAudioEncoderFactory:
        (rtc::scoped_refptr<webrtc::AudioEncoderFactory>)audioEncoderFactory
            nativeAudioDecoderFactory:
                (rtc::scoped_refptr<webrtc::AudioDecoderFactory>)audioDecoderFactory
            nativeVideoEncoderFactory:
                (std::unique_ptr<webrtc::VideoEncoderFactory>)videoEncoderFactory
            nativeVideoDecoderFactory:
                (std::unique_ptr<webrtc::VideoDecoderFactory>)videoDecoderFactory
                    audioDeviceModule:(nullable webrtc::AudioDeviceModule *)audioDeviceModule
                audioProcessingModule:
                    (rtc::scoped_refptr<webrtc::AudioProcessing>)audioProcessingModule
             networkControllerFactory:(std::unique_ptr<webrtc::NetworkControllerFactoryInterface>)
                                          networkControllerFactory
                mediaTransportFactory:
                    (std::unique_ptr<webrtc::MediaTransportFactory>)mediaTransportFactory;

- (instancetype)initWithEncoderFactory:(nullable id<RTCVideoEncoderFactory>)encoderFactory
                        decoderFactory:(nullable id<RTCVideoDecoderFactory>)decoderFactory
                 mediaTransportFactory:
                     (std::unique_ptr<webrtc::MediaTransportFactory>)mediaTransportFactory;

/** Initialize an RTCPeerConnection with a configuration, constraints, and
 *  dependencies.
 */
- (RTCPeerConnection *)
    peerConnectionWithDependencies:(RTCConfiguration *)configuration
                       constraints:(RTCMediaConstraints *)constraints
                      dependencies:(std::unique_ptr<webrtc::PeerConnectionDependencies>)dependencies
                          delegate:(nullable id<RTCPeerConnectionDelegate>)delegate;

@end

NS_ASSUME_NONNULL_END
