/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 16, 2024.
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
#import "RTCRtpTransceiver+Private.h"

#import "RTCRtpEncodingParameters+Private.h"
#import "RTCRtpParameters+Private.h"
#import "RTCRtpReceiver+Private.h"
#import "RTCRtpSender+Private.h"
#import "base/RTCLogging.h"
#import "helpers/NSString+StdString.h"

@implementation RTCRtpTransceiverInit

@synthesize direction = _direction;
@synthesize streamIds = _streamIds;
@synthesize sendEncodings = _sendEncodings;

- (instancetype)init {
  if (self = [super init]) {
    _direction = RTCRtpTransceiverDirectionSendRecv;
  }
  return self;
}

- (webrtc::RtpTransceiverInit)nativeInit {
  webrtc::RtpTransceiverInit init;
  init.direction = [RTCRtpTransceiver nativeRtpTransceiverDirectionFromDirection:_direction];
  for (NSString *streamId in _streamIds) {
    init.stream_ids.push_back([streamId UTF8String]);
  }
  for (RTCRtpEncodingParameters *sendEncoding in _sendEncodings) {
    init.send_encodings.push_back(sendEncoding.nativeParameters);
  }
  return init;
}

@end

@implementation RTCRtpTransceiver {
  RTCPeerConnectionFactory *_factory;
  rtc::scoped_refptr<webrtc::RtpTransceiverInterface> _nativeRtpTransceiver;
}

- (RTCRtpMediaType)mediaType {
  return [RTCRtpReceiver mediaTypeForNativeMediaType:_nativeRtpTransceiver->media_type()];
}

- (NSString *)mid {
  if (_nativeRtpTransceiver->mid()) {
    return [NSString stringForStdString:*_nativeRtpTransceiver->mid()];
  } else {
    return nil;
  }
}

@synthesize sender = _sender;
@synthesize receiver = _receiver;

- (BOOL)isStopped {
  return _nativeRtpTransceiver->stopped();
}

- (RTCRtpTransceiverDirection)direction {
  return [RTCRtpTransceiver
      rtpTransceiverDirectionFromNativeDirection:_nativeRtpTransceiver->direction()];
}

- (void)setDirection:(RTCRtpTransceiverDirection)direction {
  _nativeRtpTransceiver->SetDirection(
      [RTCRtpTransceiver nativeRtpTransceiverDirectionFromDirection:direction]);
}

- (BOOL)currentDirection:(RTCRtpTransceiverDirection *)currentDirectionOut {
  if (_nativeRtpTransceiver->current_direction()) {
    *currentDirectionOut = [RTCRtpTransceiver
        rtpTransceiverDirectionFromNativeDirection:*_nativeRtpTransceiver->current_direction()];
    return YES;
  } else {
    return NO;
  }
}

- (void)stop {
  _nativeRtpTransceiver->Stop();
}

- (NSString *)description {
  return [NSString
      stringWithFormat:@"RTCRtpTransceiver {\n  sender: %@\n  receiver: %@\n}", _sender, _receiver];
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
  RTCRtpTransceiver *transceiver = (RTCRtpTransceiver *)object;
  return _nativeRtpTransceiver == transceiver.nativeRtpTransceiver;
}

- (NSUInteger)hash {
  return (NSUInteger)_nativeRtpTransceiver.get();
}

#pragma mark - Private

- (rtc::scoped_refptr<webrtc::RtpTransceiverInterface>)nativeRtpTransceiver {
  return _nativeRtpTransceiver;
}

- (instancetype)initWithFactory:(RTCPeerConnectionFactory *)factory
           nativeRtpTransceiver:
               (rtc::scoped_refptr<webrtc::RtpTransceiverInterface>)nativeRtpTransceiver {
  NSParameterAssert(factory);
  NSParameterAssert(nativeRtpTransceiver);
  if (self = [super init]) {
    _factory = factory;
    _nativeRtpTransceiver = nativeRtpTransceiver;
    _sender = [[RTCRtpSender alloc] initWithFactory:_factory
                                    nativeRtpSender:nativeRtpTransceiver->sender()];
    _receiver = [[RTCRtpReceiver alloc] initWithFactory:_factory
                                      nativeRtpReceiver:nativeRtpTransceiver->receiver()];
    RTCLogInfo(@"RTCRtpTransceiver(%p): created transceiver: %@", self, self.description);
  }
  return self;
}

+ (webrtc::RtpTransceiverDirection)nativeRtpTransceiverDirectionFromDirection:
        (RTCRtpTransceiverDirection)direction {
  switch (direction) {
    case RTCRtpTransceiverDirectionSendRecv:
      return webrtc::RtpTransceiverDirection::kSendRecv;
    case RTCRtpTransceiverDirectionSendOnly:
      return webrtc::RtpTransceiverDirection::kSendOnly;
    case RTCRtpTransceiverDirectionRecvOnly:
      return webrtc::RtpTransceiverDirection::kRecvOnly;
    case RTCRtpTransceiverDirectionInactive:
      return webrtc::RtpTransceiverDirection::kInactive;
    case RTCRtpTransceiverDirectionStopped:
      return webrtc::RtpTransceiverDirection::kStopped;
  }
}

+ (RTCRtpTransceiverDirection)rtpTransceiverDirectionFromNativeDirection:
        (webrtc::RtpTransceiverDirection)nativeDirection {
  switch (nativeDirection) {
    case webrtc::RtpTransceiverDirection::kSendRecv:
      return RTCRtpTransceiverDirectionSendRecv;
    case webrtc::RtpTransceiverDirection::kSendOnly:
      return RTCRtpTransceiverDirectionSendOnly;
    case webrtc::RtpTransceiverDirection::kRecvOnly:
      return RTCRtpTransceiverDirectionRecvOnly;
    case webrtc::RtpTransceiverDirection::kInactive:
      return RTCRtpTransceiverDirectionInactive;
    case webrtc::RtpTransceiverDirection::kStopped:
      return RTCRtpTransceiverDirectionStopped;
  }
}

@end
