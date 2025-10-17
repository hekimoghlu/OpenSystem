/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 15, 2024.
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

NS_ASSUME_NONNULL_BEGIN

@class RTCAudioSource;
@class RTCAudioTrack;
@class RTCConfiguration;
@class RTCMediaConstraints;
@class RTCMediaStream;
@class RTCPeerConnection;
@class RTCVideoSource;
@class RTCVideoTrack;
@class RTCPeerConnectionFactoryOptions;
@protocol RTCPeerConnectionDelegate;
@protocol RTCVideoDecoderFactory;
@protocol RTCVideoEncoderFactory;

RTC_OBJC_EXPORT
@interface RTCPeerConnectionFactory : NSObject

/* Initialize object with default H264 video encoder/decoder factories */
- (instancetype)init;

/* Initialize object with injectable video encoder/decoder factories */
- (instancetype)initWithEncoderFactory:(nullable id<RTCVideoEncoderFactory>)encoderFactory
                        decoderFactory:(nullable id<RTCVideoDecoderFactory>)decoderFactory;

/** Initialize an RTCAudioSource with constraints. */
- (RTCAudioSource *)audioSourceWithConstraints:(nullable RTCMediaConstraints *)constraints;

/** Initialize an RTCAudioTrack with an id. Convenience ctor to use an audio source with no
 *  constraints.
 */
- (RTCAudioTrack *)audioTrackWithTrackId:(NSString *)trackId;

/** Initialize an RTCAudioTrack with a source and an id. */
- (RTCAudioTrack *)audioTrackWithSource:(RTCAudioSource *)source trackId:(NSString *)trackId;

/** Initialize a generic RTCVideoSource. The RTCVideoSource should be passed to a RTCVideoCapturer
 *  implementation, e.g. RTCCameraVideoCapturer, in order to produce frames.
 */
- (RTCVideoSource *)videoSource;

/** Initialize an RTCVideoTrack with a source and an id. */
- (RTCVideoTrack *)videoTrackWithSource:(RTCVideoSource *)source trackId:(NSString *)trackId;

/** Initialize an RTCMediaStream with an id. */
- (RTCMediaStream *)mediaStreamWithStreamId:(NSString *)streamId;

/** Initialize an RTCPeerConnection with a configuration, constraints, and
 *  delegate.
 */
- (RTCPeerConnection *)peerConnectionWithConfiguration:(RTCConfiguration *)configuration
                                           constraints:(RTCMediaConstraints *)constraints
                                              delegate:
                                                  (nullable id<RTCPeerConnectionDelegate>)delegate;

/** Set the options to be used for subsequently created RTCPeerConnections */
- (void)setOptions:(nonnull RTCPeerConnectionFactoryOptions *)options;

/** Start an AecDump recording. This API call will likely change in the future. */
- (BOOL)startAecDumpWithFilePath:(NSString *)filePath maxSizeInBytes:(int64_t)maxSizeInBytes;

/* Stop an active AecDump recording */
- (void)stopAecDump;

@end

NS_ASSUME_NONNULL_END
