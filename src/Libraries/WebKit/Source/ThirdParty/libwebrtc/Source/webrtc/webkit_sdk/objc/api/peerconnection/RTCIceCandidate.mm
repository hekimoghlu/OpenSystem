/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 9, 2022.
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
#import "RTCIceCandidate+Private.h"

#include <memory>

#import "base/RTCLogging.h"
#import "helpers/NSString+StdString.h"

@implementation RTCIceCandidate

@synthesize sdpMid = _sdpMid;
@synthesize sdpMLineIndex = _sdpMLineIndex;
@synthesize sdp = _sdp;
@synthesize serverUrl = _serverUrl;

- (instancetype)initWithSdp:(NSString *)sdp
              sdpMLineIndex:(int)sdpMLineIndex
                     sdpMid:(NSString *)sdpMid {
  NSParameterAssert(sdp.length);
  if (self = [super init]) {
    _sdpMid = [sdpMid copy];
    _sdpMLineIndex = sdpMLineIndex;
    _sdp = [sdp copy];
  }
  return self;
}

- (NSString *)description {
  return [NSString stringWithFormat:@"RTCIceCandidate:\n%@\n%d\n%@\n%@",
                                    _sdpMid,
                                    _sdpMLineIndex,
                                    _sdp,
                                    _serverUrl];
}

#pragma mark - Private

- (instancetype)initWithNativeCandidate:
    (const webrtc::IceCandidateInterface *)candidate {
  NSParameterAssert(candidate);
  std::string sdp;
  candidate->ToString(&sdp);

  RTCIceCandidate *rtcCandidate =
      [self initWithSdp:[NSString stringForStdString:sdp]
          sdpMLineIndex:candidate->sdp_mline_index()
                 sdpMid:[NSString stringForStdString:candidate->sdp_mid()]];
  rtcCandidate->_serverUrl = [NSString stringForStdString:candidate->server_url()];
  return rtcCandidate;
}

- (std::unique_ptr<webrtc::IceCandidateInterface>)nativeCandidate {
  webrtc::SdpParseError error;

  webrtc::IceCandidateInterface *candidate = webrtc::CreateIceCandidate(
      _sdpMid.stdString, _sdpMLineIndex, _sdp.stdString, &error);

  if (!candidate) {
    RTCLog(@"Failed to create ICE candidate: %s\nline: %s",
           error.description.c_str(),
           error.line.c_str());
  }

  return std::unique_ptr<webrtc::IceCandidateInterface>(candidate);
}

@end
