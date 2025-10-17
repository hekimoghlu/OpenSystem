/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 27, 2022.
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

#import "api/peerconnection/RTCIceServer+Private.h"
#import "api/peerconnection/RTCIceServer.h"
#import "helpers/NSString+StdString.h"

@interface RTCIceServerTest : NSObject
- (void)testOneURLServer;
- (void)testTwoURLServer;
- (void)testPasswordCredential;
- (void)testInitFromNativeServer;
@end

@implementation RTCIceServerTest

- (void)testOneURLServer {
  RTCIceServer *server = [[RTCIceServer alloc] initWithURLStrings:@[
      @"stun:stun1.example.net" ]];

  webrtc::PeerConnectionInterface::IceServer iceStruct = server.nativeServer;
  EXPECT_EQ(1u, iceStruct.urls.size());
  EXPECT_EQ("stun:stun1.example.net", iceStruct.urls.front());
  EXPECT_EQ("", iceStruct.username);
  EXPECT_EQ("", iceStruct.password);
}

- (void)testTwoURLServer {
  RTCIceServer *server = [[RTCIceServer alloc] initWithURLStrings:@[
      @"turn1:turn1.example.net", @"turn2:turn2.example.net" ]];

  webrtc::PeerConnectionInterface::IceServer iceStruct = server.nativeServer;
  EXPECT_EQ(2u, iceStruct.urls.size());
  EXPECT_EQ("turn1:turn1.example.net", iceStruct.urls.front());
  EXPECT_EQ("turn2:turn2.example.net", iceStruct.urls.back());
  EXPECT_EQ("", iceStruct.username);
  EXPECT_EQ("", iceStruct.password);
}

- (void)testPasswordCredential {
  RTCIceServer *server = [[RTCIceServer alloc]
      initWithURLStrings:@[ @"turn1:turn1.example.net" ]
                username:@"username"
              credential:@"credential"];
  webrtc::PeerConnectionInterface::IceServer iceStruct = server.nativeServer;
  EXPECT_EQ(1u, iceStruct.urls.size());
  EXPECT_EQ("turn1:turn1.example.net", iceStruct.urls.front());
  EXPECT_EQ("username", iceStruct.username);
  EXPECT_EQ("credential", iceStruct.password);
}

- (void)testHostname {
  RTCIceServer *server = [[RTCIceServer alloc] initWithURLStrings:@[ @"turn1:turn1.example.net" ]
                                                         username:@"username"
                                                       credential:@"credential"
                                                    tlsCertPolicy:RTCTlsCertPolicySecure
                                                         hostname:@"hostname"];
  webrtc::PeerConnectionInterface::IceServer iceStruct = server.nativeServer;
  EXPECT_EQ(1u, iceStruct.urls.size());
  EXPECT_EQ("turn1:turn1.example.net", iceStruct.urls.front());
  EXPECT_EQ("username", iceStruct.username);
  EXPECT_EQ("credential", iceStruct.password);
  EXPECT_EQ("hostname", iceStruct.hostname);
}

- (void)testTlsAlpnProtocols {
  RTCIceServer *server = [[RTCIceServer alloc] initWithURLStrings:@[ @"turn1:turn1.example.net" ]
                                                         username:@"username"
                                                       credential:@"credential"
                                                    tlsCertPolicy:RTCTlsCertPolicySecure
                                                         hostname:@"hostname"
                                                 tlsAlpnProtocols:@[ @"proto1", @"proto2" ]];
  webrtc::PeerConnectionInterface::IceServer iceStruct = server.nativeServer;
  EXPECT_EQ(1u, iceStruct.urls.size());
  EXPECT_EQ("turn1:turn1.example.net", iceStruct.urls.front());
  EXPECT_EQ("username", iceStruct.username);
  EXPECT_EQ("credential", iceStruct.password);
  EXPECT_EQ("hostname", iceStruct.hostname);
  EXPECT_EQ(2u, iceStruct.tls_alpn_protocols.size());
}

- (void)testTlsEllipticCurves {
  RTCIceServer *server = [[RTCIceServer alloc] initWithURLStrings:@[ @"turn1:turn1.example.net" ]
                                                         username:@"username"
                                                       credential:@"credential"
                                                    tlsCertPolicy:RTCTlsCertPolicySecure
                                                         hostname:@"hostname"
                                                 tlsAlpnProtocols:@[ @"proto1", @"proto2" ]
                                                tlsEllipticCurves:@[ @"curve1", @"curve2" ]];
  webrtc::PeerConnectionInterface::IceServer iceStruct = server.nativeServer;
  EXPECT_EQ(1u, iceStruct.urls.size());
  EXPECT_EQ("turn1:turn1.example.net", iceStruct.urls.front());
  EXPECT_EQ("username", iceStruct.username);
  EXPECT_EQ("credential", iceStruct.password);
  EXPECT_EQ("hostname", iceStruct.hostname);
  EXPECT_EQ(2u, iceStruct.tls_alpn_protocols.size());
  EXPECT_EQ(2u, iceStruct.tls_elliptic_curves.size());
}

- (void)testInitFromNativeServer {
  webrtc::PeerConnectionInterface::IceServer nativeServer;
  nativeServer.username = "username";
  nativeServer.password = "password";
  nativeServer.urls.push_back("stun:stun.example.net");
  nativeServer.hostname = "hostname";
  nativeServer.tls_alpn_protocols.push_back("proto1");
  nativeServer.tls_alpn_protocols.push_back("proto2");
  nativeServer.tls_elliptic_curves.push_back("curve1");
  nativeServer.tls_elliptic_curves.push_back("curve2");

  RTCIceServer *iceServer =
      [[RTCIceServer alloc] initWithNativeServer:nativeServer];
  EXPECT_EQ(1u, iceServer.urlStrings.count);
  EXPECT_EQ("stun:stun.example.net",
      [NSString stdStringForString:iceServer.urlStrings.firstObject]);
  EXPECT_EQ("username", [NSString stdStringForString:iceServer.username]);
  EXPECT_EQ("password", [NSString stdStringForString:iceServer.credential]);
  EXPECT_EQ("hostname", [NSString stdStringForString:iceServer.hostname]);
  EXPECT_EQ(2u, iceServer.tlsAlpnProtocols.count);
  EXPECT_EQ(2u, iceServer.tlsEllipticCurves.count);
}

@end

TEST(RTCIceServerTest, OneURLTest) {
  @autoreleasepool {
    RTCIceServerTest *test = [[RTCIceServerTest alloc] init];
    [test testOneURLServer];
  }
}

TEST(RTCIceServerTest, TwoURLTest) {
  @autoreleasepool {
    RTCIceServerTest *test = [[RTCIceServerTest alloc] init];
    [test testTwoURLServer];
  }
}

TEST(RTCIceServerTest, PasswordCredentialTest) {
  @autoreleasepool {
    RTCIceServerTest *test = [[RTCIceServerTest alloc] init];
    [test testPasswordCredential];
  }
}

TEST(RTCIceServerTest, HostnameTest) {
  @autoreleasepool {
    RTCIceServerTest *test = [[RTCIceServerTest alloc] init];
    [test testHostname];
  }
}

TEST(RTCIceServerTest, InitFromNativeServerTest) {
  @autoreleasepool {
    RTCIceServerTest *test = [[RTCIceServerTest alloc] init];
    [test testInitFromNativeServer];
  }
}
