/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 18, 2023.
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

#include "rtc_base/gunit.h"

#import "api/peerconnection/RTCDataChannelConfiguration+Private.h"
#import "api/peerconnection/RTCDataChannelConfiguration.h"
#import "helpers/NSString+StdString.h"

@interface RTCDataChannelConfigurationTest : NSObject
- (void)testConversionToNativeDataChannelInit;
@end

@implementation RTCDataChannelConfigurationTest

- (void)testConversionToNativeDataChannelInit {
  BOOL isOrdered = NO;
  int maxPacketLifeTime = 5;
  int maxRetransmits = 4;
  BOOL isNegotiated = YES;
  int channelId = 4;
  NSString *protocol = @"protocol";

  RTCDataChannelConfiguration *dataChannelConfig =
      [[RTCDataChannelConfiguration alloc] init];
  dataChannelConfig.isOrdered = isOrdered;
  dataChannelConfig.maxPacketLifeTime = maxPacketLifeTime;
  dataChannelConfig.maxRetransmits = maxRetransmits;
  dataChannelConfig.isNegotiated = isNegotiated;
  dataChannelConfig.channelId = channelId;
  dataChannelConfig.protocol = protocol;

  webrtc::DataChannelInit nativeInit = dataChannelConfig.nativeDataChannelInit;
  EXPECT_EQ(isOrdered, nativeInit.ordered);
  EXPECT_EQ(maxPacketLifeTime, nativeInit.maxRetransmitTime);
  EXPECT_EQ(maxRetransmits, nativeInit.maxRetransmits);
  EXPECT_EQ(isNegotiated, nativeInit.negotiated);
  EXPECT_EQ(channelId, nativeInit.id);
  EXPECT_EQ(protocol.stdString, nativeInit.protocol);
}

@end

TEST(RTCDataChannelConfiguration, NativeDataChannelInitConversionTest) {
  @autoreleasepool {
    RTCDataChannelConfigurationTest *test =
        [[RTCDataChannelConfigurationTest alloc] init];
    [test testConversionToNativeDataChannelInit];
  }
}
