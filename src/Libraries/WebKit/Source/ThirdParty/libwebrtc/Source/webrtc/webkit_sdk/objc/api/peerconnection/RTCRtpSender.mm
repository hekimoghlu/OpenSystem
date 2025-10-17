/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 2, 2023.
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
#import "RTCRtpSender+Private.h"

#import "RTCDtmfSender+Private.h"
#import "RTCMediaStreamTrack+Private.h"
#import "RTCRtpParameters+Private.h"
#import "RTCRtpSender+Native.h"
#import "base/RTCLogging.h"
#import "helpers/NSString+StdString.h"

#include "api/media_stream_interface.h"

@implementation RTCRtpSender {
  RTCPeerConnectionFactory *_factory;
  rtc::scoped_refptr<webrtc::RtpSenderInterface> _nativeRtpSender;
}

@synthesize dtmfSender = _dtmfSender;

- (NSString *)senderId {
  return [NSString stringForStdString:_nativeRtpSender->id()];
}

- (RTCRtpParameters *)parameters {
  return [[RTCRtpParameters alloc]
      initWithNativeParameters:_nativeRtpSender->GetParameters()];
}

- (void)setParameters:(RTCRtpParameters *)parameters {
  if (!_nativeRtpSender->SetParameters(parameters.nativeParameters).ok()) {
    RTCLogError(@"RTCRtpSender(%p): Failed to set parameters: %@", self,
        parameters);
  }
}

- (RTCMediaStreamTrack *)track {
  rtc::scoped_refptr<webrtc::MediaStreamTrackInterface> nativeTrack(
    _nativeRtpSender->track());
  if (nativeTrack) {
    return [RTCMediaStreamTrack mediaTrackForNativeTrack:nativeTrack factory:_factory];
  }
  return nil;
}

- (void)setTrack:(RTCMediaStreamTrack *)track {
  if (!_nativeRtpSender->SetTrack(track.nativeTrack)) {
    RTCLogError(@"RTCRtpSender(%p): Failed to set track %@", self, track);
  }
}

- (NSArray<NSString *> *)streamIds {
  std::vector<std::string> nativeStreamIds = _nativeRtpSender->stream_ids();
  NSMutableArray *streamIds = [NSMutableArray arrayWithCapacity:nativeStreamIds.size()];
  for (const auto &s : nativeStreamIds) {
    [streamIds addObject:[NSString stringForStdString:s]];
  }
  return streamIds;
}

- (void)setStreamIds:(NSArray<NSString *> *)streamIds {
  std::vector<std::string> nativeStreamIds;
  for (NSString *streamId in streamIds) {
    nativeStreamIds.push_back([streamId UTF8String]);
  }
  _nativeRtpSender->SetStreams(nativeStreamIds);
}

- (NSString *)description {
  return [NSString stringWithFormat:@"RTCRtpSender {\n  senderId: %@\n}",
      self.senderId];
}

- (BOOL)isEqual:(id)object {
  if (self == object) {
    return YES;
  }
  if (object == nil) {
    return NO;
  }
  if (![object isMemberOfClass:[self class]]) {
    return NO;
  }
  RTCRtpSender *sender = (RTCRtpSender *)object;
  return _nativeRtpSender == sender.nativeRtpSender;
}

- (NSUInteger)hash {
  return (NSUInteger)_nativeRtpSender.get();
}

#pragma mark - Native

- (void)setFrameEncryptor:(rtc::scoped_refptr<webrtc::FrameEncryptorInterface>)frameEncryptor {
  _nativeRtpSender->SetFrameEncryptor(frameEncryptor);
}

#pragma mark - Private

- (rtc::scoped_refptr<webrtc::RtpSenderInterface>)nativeRtpSender {
  return _nativeRtpSender;
}

- (instancetype)initWithFactory:(RTCPeerConnectionFactory *)factory
                nativeRtpSender:(rtc::scoped_refptr<webrtc::RtpSenderInterface>)nativeRtpSender {
  NSParameterAssert(factory);
  NSParameterAssert(nativeRtpSender);
  if (self = [super init]) {
    _factory = factory;
    _nativeRtpSender = nativeRtpSender;
    rtc::scoped_refptr<webrtc::DtmfSenderInterface> nativeDtmfSender(
        _nativeRtpSender->GetDtmfSender());
    if (nativeDtmfSender) {
      _dtmfSender = [[RTCDtmfSender alloc] initWithNativeDtmfSender:nativeDtmfSender];
    }
    RTCLogInfo(@"RTCRtpSender(%p): created sender: %@", self, self.description);
  }
  return self;
}

@end
