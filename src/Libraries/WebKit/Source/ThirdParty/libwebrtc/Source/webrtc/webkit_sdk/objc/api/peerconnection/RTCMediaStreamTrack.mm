/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 26, 2023.
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
#import "RTCAudioTrack+Private.h"
#import "RTCMediaStreamTrack+Private.h"
#import "RTCVideoTrack+Private.h"

#import "helpers/NSString+StdString.h"

NSString * const kRTCMediaStreamTrackKindAudio =
    @(webrtc::MediaStreamTrackInterface::kAudioKind);
NSString * const kRTCMediaStreamTrackKindVideo =
    @(webrtc::MediaStreamTrackInterface::kVideoKind);

@implementation RTCMediaStreamTrack {
  RTCPeerConnectionFactory *_factory;
  rtc::scoped_refptr<webrtc::MediaStreamTrackInterface> _nativeTrack;
  RTCMediaStreamTrackType _type;
}

- (NSString *)kind {
  return [NSString stringForStdString:_nativeTrack->kind()];
}

- (NSString *)trackId {
  return [NSString stringForStdString:_nativeTrack->id()];
}

- (BOOL)isEnabled {
  return _nativeTrack->enabled();
}

- (void)setIsEnabled:(BOOL)isEnabled {
  _nativeTrack->set_enabled(isEnabled);
}

- (RTCMediaStreamTrackState)readyState {
  return [[self class] trackStateForNativeState:_nativeTrack->state()];
}

- (NSString *)description {
  NSString *readyState = [[self class] stringForState:self.readyState];
  return [NSString stringWithFormat:@"RTCMediaStreamTrack:\n%@\n%@\n%@\n%@",
                                    self.kind,
                                    self.trackId,
                                    self.isEnabled ? @"enabled" : @"disabled",
                                    readyState];
}

- (BOOL)isEqual:(id)object {
  if (self == object) {
    return YES;
  }
  if (![object isMemberOfClass:[self class]]) {
    return NO;
  }
  return [self isEqualToTrack:(RTCMediaStreamTrack *)object];
}

- (NSUInteger)hash {
  return (NSUInteger)_nativeTrack.get();
}

#pragma mark - Private

- (rtc::scoped_refptr<webrtc::MediaStreamTrackInterface>)nativeTrack {
  return _nativeTrack;
}

@synthesize factory = _factory;

- (instancetype)initWithFactory:(RTCPeerConnectionFactory *)factory
                    nativeTrack:(rtc::scoped_refptr<webrtc::MediaStreamTrackInterface>)nativeTrack
                           type:(RTCMediaStreamTrackType)type {
  NSParameterAssert(nativeTrack);
  NSParameterAssert(factory);
  if (self = [super init]) {
    _factory = factory;
    _nativeTrack = nativeTrack;
    _type = type;
  }
  return self;
}

- (instancetype)initWithFactory:(RTCPeerConnectionFactory *)factory
                    nativeTrack:(rtc::scoped_refptr<webrtc::MediaStreamTrackInterface>)nativeTrack {
  NSParameterAssert(nativeTrack);
  if (nativeTrack->kind() ==
      std::string(webrtc::MediaStreamTrackInterface::kAudioKind)) {
    return [self initWithFactory:factory nativeTrack:nativeTrack type:RTCMediaStreamTrackTypeAudio];
  }
  if (nativeTrack->kind() ==
      std::string(webrtc::MediaStreamTrackInterface::kVideoKind)) {
    return [self initWithFactory:factory nativeTrack:nativeTrack type:RTCMediaStreamTrackTypeVideo];
  }
  return nil;
}

- (BOOL)isEqualToTrack:(RTCMediaStreamTrack *)track {
  if (!track) {
    return NO;
  }
  return _nativeTrack == track.nativeTrack;
}

+ (webrtc::MediaStreamTrackInterface::TrackState)nativeTrackStateForState:
    (RTCMediaStreamTrackState)state {
  switch (state) {
    case RTCMediaStreamTrackStateLive:
      return webrtc::MediaStreamTrackInterface::kLive;
    case RTCMediaStreamTrackStateEnded:
      return webrtc::MediaStreamTrackInterface::kEnded;
  }
}

+ (RTCMediaStreamTrackState)trackStateForNativeState:
    (webrtc::MediaStreamTrackInterface::TrackState)nativeState {
  switch (nativeState) {
    case webrtc::MediaStreamTrackInterface::kLive:
      return RTCMediaStreamTrackStateLive;
    case webrtc::MediaStreamTrackInterface::kEnded:
      return RTCMediaStreamTrackStateEnded;
  }
}

+ (NSString *)stringForState:(RTCMediaStreamTrackState)state {
  switch (state) {
    case RTCMediaStreamTrackStateLive:
      return @"Live";
    case RTCMediaStreamTrackStateEnded:
      return @"Ended";
  }
}

+ (RTCMediaStreamTrack *)mediaTrackForNativeTrack:
                             (rtc::scoped_refptr<webrtc::MediaStreamTrackInterface>)nativeTrack
                                          factory:(RTCPeerConnectionFactory *)factory {
  NSParameterAssert(nativeTrack);
  NSParameterAssert(factory);
  if (nativeTrack->kind() == webrtc::MediaStreamTrackInterface::kAudioKind) {
    return [[RTCAudioTrack alloc] initWithFactory:factory
                                      nativeTrack:nativeTrack
                                             type:RTCMediaStreamTrackTypeAudio];
  } else if (nativeTrack->kind() == webrtc::MediaStreamTrackInterface::kVideoKind) {
    return [[RTCVideoTrack alloc] initWithFactory:factory
                                      nativeTrack:nativeTrack
                                             type:RTCMediaStreamTrackTypeVideo];
  } else {
    return [[RTCMediaStreamTrack alloc] initWithFactory:factory nativeTrack:nativeTrack];
  }
}

@end
