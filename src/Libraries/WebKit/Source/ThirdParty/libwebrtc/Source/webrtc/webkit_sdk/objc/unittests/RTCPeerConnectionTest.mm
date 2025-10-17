/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 16, 2025.
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
#include <vector>

#include "rtc_base/gunit.h"

#import "api/peerconnection/RTCConfiguration+Private.h"
#import "api/peerconnection/RTCConfiguration.h"
#import "api/peerconnection/RTCCryptoOptions.h"
#import "api/peerconnection/RTCIceServer.h"
#import "api/peerconnection/RTCMediaConstraints.h"
#import "api/peerconnection/RTCPeerConnection.h"
#import "api/peerconnection/RTCPeerConnectionFactory+Native.h"
#import "api/peerconnection/RTCPeerConnectionFactory.h"
#import "helpers/NSString+StdString.h"

@interface RTCPeerConnectionTest : NSObject
- (void)testConfigurationGetter;
- (void)testWithDependencies;
@end

@implementation RTCPeerConnectionTest

- (void)testConfigurationGetter {
  NSArray *urlStrings = @[ @"stun:stun1.example.net" ];
  RTCIceServer *server = [[RTCIceServer alloc] initWithURLStrings:urlStrings];

  RTCConfiguration *config = [[RTCConfiguration alloc] init];
  config.iceServers = @[ server ];
  config.iceTransportPolicy = RTCIceTransportPolicyRelay;
  config.bundlePolicy = RTCBundlePolicyMaxBundle;
  config.rtcpMuxPolicy = RTCRtcpMuxPolicyNegotiate;
  config.tcpCandidatePolicy = RTCTcpCandidatePolicyDisabled;
  config.candidateNetworkPolicy = RTCCandidateNetworkPolicyLowCost;
  const int maxPackets = 60;
  const int timeout = 1500;
  const int interval = 2000;
  config.audioJitterBufferMaxPackets = maxPackets;
  config.audioJitterBufferFastAccelerate = YES;
  config.iceConnectionReceivingTimeout = timeout;
  config.iceBackupCandidatePairPingInterval = interval;
  config.continualGatheringPolicy =
      RTCContinualGatheringPolicyGatherContinually;
  config.shouldPruneTurnPorts = YES;
  config.activeResetSrtpParams = YES;
  config.cryptoOptions = [[RTCCryptoOptions alloc] initWithSrtpEnableGcmCryptoSuites:YES
                                                 srtpEnableAes128Sha1_32CryptoCipher:YES
                                              srtpEnableEncryptedRtpHeaderExtensions:NO
                                                        sframeRequireFrameEncryption:NO];

  RTCMediaConstraints *contraints = [[RTCMediaConstraints alloc] initWithMandatoryConstraints:@{}
      optionalConstraints:nil];
  RTCPeerConnectionFactory *factory = [[RTCPeerConnectionFactory alloc] init];

  RTCConfiguration *newConfig;
  @autoreleasepool {
    RTCPeerConnection *peerConnection =
        [factory peerConnectionWithConfiguration:config constraints:contraints delegate:nil];
    newConfig = peerConnection.configuration;

    EXPECT_TRUE([peerConnection setBweMinBitrateBps:[NSNumber numberWithInt:100000]
                                  currentBitrateBps:[NSNumber numberWithInt:5000000]
                                      maxBitrateBps:[NSNumber numberWithInt:500000000]]);
    EXPECT_FALSE([peerConnection setBweMinBitrateBps:[NSNumber numberWithInt:2]
                                   currentBitrateBps:[NSNumber numberWithInt:1]
                                       maxBitrateBps:nil]);
  }

  EXPECT_EQ([config.iceServers count], [newConfig.iceServers count]);
  RTCIceServer *newServer = newConfig.iceServers[0];
  RTCIceServer *origServer = config.iceServers[0];
  std::string origUrl = origServer.urlStrings.firstObject.UTF8String;
  std::string url = newServer.urlStrings.firstObject.UTF8String;
  EXPECT_EQ(origUrl, url);

  EXPECT_EQ(config.iceTransportPolicy, newConfig.iceTransportPolicy);
  EXPECT_EQ(config.bundlePolicy, newConfig.bundlePolicy);
  EXPECT_EQ(config.rtcpMuxPolicy, newConfig.rtcpMuxPolicy);
  EXPECT_EQ(config.tcpCandidatePolicy, newConfig.tcpCandidatePolicy);
  EXPECT_EQ(config.candidateNetworkPolicy, newConfig.candidateNetworkPolicy);
  EXPECT_EQ(config.audioJitterBufferMaxPackets, newConfig.audioJitterBufferMaxPackets);
  EXPECT_EQ(config.audioJitterBufferFastAccelerate, newConfig.audioJitterBufferFastAccelerate);
  EXPECT_EQ(config.iceConnectionReceivingTimeout, newConfig.iceConnectionReceivingTimeout);
  EXPECT_EQ(config.iceBackupCandidatePairPingInterval,
            newConfig.iceBackupCandidatePairPingInterval);
  EXPECT_EQ(config.continualGatheringPolicy, newConfig.continualGatheringPolicy);
  EXPECT_EQ(config.shouldPruneTurnPorts, newConfig.shouldPruneTurnPorts);
  EXPECT_EQ(config.activeResetSrtpParams, newConfig.activeResetSrtpParams);
  EXPECT_EQ(config.cryptoOptions.srtpEnableGcmCryptoSuites,
            newConfig.cryptoOptions.srtpEnableGcmCryptoSuites);
  EXPECT_EQ(config.cryptoOptions.srtpEnableAes128Sha1_32CryptoCipher,
            newConfig.cryptoOptions.srtpEnableAes128Sha1_32CryptoCipher);
  EXPECT_EQ(config.cryptoOptions.srtpEnableEncryptedRtpHeaderExtensions,
            newConfig.cryptoOptions.srtpEnableEncryptedRtpHeaderExtensions);
  EXPECT_EQ(config.cryptoOptions.sframeRequireFrameEncryption,
            newConfig.cryptoOptions.sframeRequireFrameEncryption);
}

- (void)testWithDependencies {
  NSArray *urlStrings = @[ @"stun:stun1.example.net" ];
  RTCIceServer *server = [[RTCIceServer alloc] initWithURLStrings:urlStrings];

  RTCConfiguration *config = [[RTCConfiguration alloc] init];
  config.iceServers = @[ server ];
  RTCMediaConstraints *contraints = [[RTCMediaConstraints alloc] initWithMandatoryConstraints:@{}
                                                                          optionalConstraints:nil];
  RTCPeerConnectionFactory *factory = [[RTCPeerConnectionFactory alloc] init];

  RTCConfiguration *newConfig;
  std::unique_ptr<webrtc::PeerConnectionDependencies> pc_dependencies =
      std::make_unique<webrtc::PeerConnectionDependencies>(nullptr);
  @autoreleasepool {
    RTCPeerConnection *peerConnection =
        [factory peerConnectionWithDependencies:config
                                    constraints:contraints
                                   dependencies:std::move(pc_dependencies)
                                       delegate:nil];
    newConfig = peerConnection.configuration;
  }
}

@end

TEST(RTCPeerConnectionTest, ConfigurationGetterTest) {
  @autoreleasepool {
    RTCPeerConnectionTest *test = [[RTCPeerConnectionTest alloc] init];
    [test testConfigurationGetter];
  }
}

TEST(RTCPeerConnectionTest, TestWithDependencies) {
  @autoreleasepool {
    RTCPeerConnectionTest *test = [[RTCPeerConnectionTest alloc] init];
    [test testWithDependencies];
  }
}
