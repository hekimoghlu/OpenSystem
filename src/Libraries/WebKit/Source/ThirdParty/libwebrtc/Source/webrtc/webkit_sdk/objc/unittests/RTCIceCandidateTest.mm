/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 24, 2023.
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

#include <memory>

#include "rtc_base/gunit.h"

#import "api/peerconnection/RTCIceCandidate+Private.h"
#import "api/peerconnection/RTCIceCandidate.h"
#import "helpers/NSString+StdString.h"

@interface RTCIceCandidateTest : NSObject
- (void)testCandidate;
- (void)testInitFromNativeCandidate;
@end

@implementation RTCIceCandidateTest

- (void)testCandidate {
  NSString *sdp = @"candidate:4025901590 1 udp 2122265343 "
                   "fdff:2642:12a6:fe38:c001:beda:fcf9:51aa "
                   "59052 typ host generation 0";

  RTCIceCandidate *candidate = [[RTCIceCandidate alloc] initWithSdp:sdp
                                                      sdpMLineIndex:0
                                                             sdpMid:@"audio"];

  std::unique_ptr<webrtc::IceCandidateInterface> nativeCandidate =
      candidate.nativeCandidate;
  EXPECT_EQ("audio", nativeCandidate->sdp_mid());
  EXPECT_EQ(0, nativeCandidate->sdp_mline_index());

  std::string sdpString;
  nativeCandidate->ToString(&sdpString);
  EXPECT_EQ(sdp.stdString, sdpString);
}

- (void)testInitFromNativeCandidate {
  std::string sdp("candidate:4025901590 1 udp 2122265343 "
                  "fdff:2642:12a6:fe38:c001:beda:fcf9:51aa "
                  "59052 typ host generation 0");
  webrtc::IceCandidateInterface *nativeCandidate =
      webrtc::CreateIceCandidate("audio", 0, sdp, nullptr);

  RTCIceCandidate *iceCandidate =
      [[RTCIceCandidate alloc] initWithNativeCandidate:nativeCandidate];
  EXPECT_TRUE([@"audio" isEqualToString:iceCandidate.sdpMid]);
  EXPECT_EQ(0, iceCandidate.sdpMLineIndex);

  EXPECT_EQ(sdp, iceCandidate.sdp.stdString);
}

@end

TEST(RTCIceCandidateTest, CandidateTest) {
  @autoreleasepool {
    RTCIceCandidateTest *test = [[RTCIceCandidateTest alloc] init];
    [test testCandidate];
  }
}

TEST(RTCIceCandidateTest, InitFromCandidateTest) {
  @autoreleasepool {
    RTCIceCandidateTest *test = [[RTCIceCandidateTest alloc] init];
    [test testInitFromNativeCandidate];
  }
}
