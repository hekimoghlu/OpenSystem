/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 28, 2023.
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

#include <vector>

#include "rtc_base/gunit.h"

#import "api/peerconnection/RTCConfiguration+Private.h"
#import "api/peerconnection/RTCConfiguration.h"
#import "api/peerconnection/RTCIceServer.h"
#import "api/peerconnection/RTCMediaConstraints.h"
#import "api/peerconnection/RTCPeerConnection.h"
#import "api/peerconnection/RTCPeerConnectionFactory.h"
#import "helpers/NSString+StdString.h"

@interface RTCCertificateTest : NSObject
- (void)testCertificateIsUsedInConfig;
@end

@implementation RTCCertificateTest

- (void)testCertificateIsUsedInConfig {
  RTCConfiguration *originalConfig = [[RTCConfiguration alloc] init];

  NSArray *urlStrings = @[ @"stun:stun1.example.net" ];
  RTCIceServer *server = [[RTCIceServer alloc] initWithURLStrings:urlStrings];
  originalConfig.iceServers = @[ server ];

  // Generate a new certificate.
  RTCCertificate *originalCertificate = [RTCCertificate generateCertificateWithParams:@{
    @"expires" : @100000,
    @"name" : @"RSASSA-PKCS1-v1_5"
  }];

  // Store certificate in configuration.
  originalConfig.certificate = originalCertificate;

  RTCMediaConstraints *contraints =
      [[RTCMediaConstraints alloc] initWithMandatoryConstraints:@{} optionalConstraints:nil];
  RTCPeerConnectionFactory *factory = [[RTCPeerConnectionFactory alloc] init];

  // Create PeerConnection with this certificate.
  RTCPeerConnection *peerConnection =
      [factory peerConnectionWithConfiguration:originalConfig constraints:contraints delegate:nil];

  // Retrieve certificate from the configuration.
  RTCConfiguration *retrievedConfig = peerConnection.configuration;

  // Extract PEM strings from original certificate.
  std::string originalPrivateKeyField = [[originalCertificate private_key] UTF8String];
  std::string originalCertificateField = [[originalCertificate certificate] UTF8String];

  // Extract PEM strings from certificate retrieved from configuration.
  RTCCertificate *retrievedCertificate = retrievedConfig.certificate;
  std::string retrievedPrivateKeyField = [[retrievedCertificate private_key] UTF8String];
  std::string retrievedCertificateField = [[retrievedCertificate certificate] UTF8String];

  // Check that the original certificate and retrieved certificate match.
  EXPECT_EQ(originalPrivateKeyField, retrievedPrivateKeyField);
  EXPECT_EQ(retrievedCertificateField, retrievedCertificateField);
}

@end

TEST(CertificateTest, DISABLED_CertificateIsUsedInConfig) {
  RTCCertificateTest *test = [[RTCCertificateTest alloc] init];
  [test testCertificateIsUsedInConfig];
}
