/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 12, 2023.
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
#import "RTCMediaStream+Private.h"

#include <vector>

#import "RTCAudioTrack+Private.h"
#import "RTCMediaStreamTrack+Private.h"
#import "RTCPeerConnectionFactory+Private.h"
#import "RTCVideoTrack+Private.h"
#import "helpers/NSString+StdString.h"

@implementation RTCMediaStream {
  RTCPeerConnectionFactory *_factory;
  NSMutableArray *_audioTracks;
  NSMutableArray *_videoTracks;
  rtc::scoped_refptr<webrtc::MediaStreamInterface> _nativeMediaStream;
}

- (instancetype)initWithFactory:(RTCPeerConnectionFactory *)factory
                       streamId:(NSString *)streamId {
  NSParameterAssert(factory);
  NSParameterAssert(streamId.length);
  std::string nativeId = [NSString stdStringForString:streamId];
  rtc::scoped_refptr<webrtc::MediaStreamInterface> stream =
      factory.nativeFactory->CreateLocalMediaStream(nativeId);
  return [self initWithFactory:factory nativeMediaStream:stream];
}

- (NSArray<RTCAudioTrack *> *)audioTracks {
  return [_audioTracks copy];
}

- (NSArray<RTCVideoTrack *> *)videoTracks {
  return [_videoTracks copy];
}

- (NSString *)streamId {
  return [NSString stringForStdString:_nativeMediaStream->id()];
}

- (void)addAudioTrack:(RTCAudioTrack *)audioTrack {
  if (_nativeMediaStream->AddTrack(audioTrack.nativeAudioTrack)) {
    [_audioTracks addObject:audioTrack];
  }
}

- (void)addVideoTrack:(RTCVideoTrack *)videoTrack {
  if (_nativeMediaStream->AddTrack(videoTrack.nativeVideoTrack)) {
    [_videoTracks addObject:videoTrack];
  }
}

- (void)removeAudioTrack:(RTCAudioTrack *)audioTrack {
  NSUInteger index = [_audioTracks indexOfObjectIdenticalTo:audioTrack];
  NSAssert(index != NSNotFound,
           @"|removeAudioTrack| called on unexpected RTCAudioTrack");
  if (index != NSNotFound &&
      _nativeMediaStream->RemoveTrack(audioTrack.nativeAudioTrack)) {
    [_audioTracks removeObjectAtIndex:index];
  }
}

- (void)removeVideoTrack:(RTCVideoTrack *)videoTrack {
  NSUInteger index = [_videoTracks indexOfObjectIdenticalTo:videoTrack];
  NSAssert(index != NSNotFound,
           @"|removeVideoTrack| called on unexpected RTCVideoTrack");
  if (index != NSNotFound &&
      _nativeMediaStream->RemoveTrack(videoTrack.nativeVideoTrack)) {
    [_videoTracks removeObjectAtIndex:index];
  }
}

- (NSString *)description {
  return [NSString stringWithFormat:@"RTCMediaStream:\n%@\nA=%lu\nV=%lu",
                                    self.streamId,
                                    (unsigned long)self.audioTracks.count,
                                    (unsigned long)self.videoTracks.count];
}

#pragma mark - Private

- (rtc::scoped_refptr<webrtc::MediaStreamInterface>)nativeMediaStream {
  return _nativeMediaStream;
}

- (instancetype)initWithFactory:(RTCPeerConnectionFactory *)factory
              nativeMediaStream:
                  (rtc::scoped_refptr<webrtc::MediaStreamInterface>)nativeMediaStream {
  NSParameterAssert(nativeMediaStream);
  if (self = [super init]) {
    _factory = factory;

    webrtc::AudioTrackVector audioTracks = nativeMediaStream->GetAudioTracks();
    webrtc::VideoTrackVector videoTracks = nativeMediaStream->GetVideoTracks();

    _audioTracks = [NSMutableArray arrayWithCapacity:audioTracks.size()];
    _videoTracks = [NSMutableArray arrayWithCapacity:videoTracks.size()];
    _nativeMediaStream = nativeMediaStream;

    for (auto &track : audioTracks) {
      RTCMediaStreamTrackType type = RTCMediaStreamTrackTypeAudio;
      RTCAudioTrack *audioTrack =
          [[RTCAudioTrack alloc] initWithFactory:_factory nativeTrack:track type:type];
      [_audioTracks addObject:audioTrack];
    }

    for (auto &track : videoTracks) {
      RTCMediaStreamTrackType type = RTCMediaStreamTrackTypeVideo;
      RTCVideoTrack *videoTrack =
          [[RTCVideoTrack alloc] initWithFactory:_factory nativeTrack:track type:type];
      [_videoTracks addObject:videoTrack];
    }
  }
  return self;
}

@end
