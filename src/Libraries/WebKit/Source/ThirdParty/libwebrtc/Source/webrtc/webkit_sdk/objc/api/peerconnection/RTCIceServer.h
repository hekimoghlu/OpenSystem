/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 12, 2022.
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

#import "RTCMacros.h"

typedef NS_ENUM(NSUInteger, RTCTlsCertPolicy) {
  RTCTlsCertPolicySecure,
  RTCTlsCertPolicyInsecureNoCheck
};

NS_ASSUME_NONNULL_BEGIN

RTC_OBJC_EXPORT
@interface RTCIceServer : NSObject

/** URI(s) for this server represented as NSStrings. */
@property(nonatomic, readonly) NSArray<NSString *> *urlStrings;

/** Username to use if this RTCIceServer object is a TURN server. */
@property(nonatomic, readonly, nullable) NSString *username;

/** Credential to use if this RTCIceServer object is a TURN server. */
@property(nonatomic, readonly, nullable) NSString *credential;

/**
 * TLS certificate policy to use if this RTCIceServer object is a TURN server.
 */
@property(nonatomic, readonly) RTCTlsCertPolicy tlsCertPolicy;

/**
  If the URIs in |urls| only contain IP addresses, this field can be used
  to indicate the hostname, which may be necessary for TLS (using the SNI
  extension). If |urls| itself contains the hostname, this isn't necessary.
 */
@property(nonatomic, readonly, nullable) NSString *hostname;

/** List of protocols to be used in the TLS ALPN extension. */
@property(nonatomic, readonly) NSArray<NSString *> *tlsAlpnProtocols;

/**
  List elliptic curves to be used in the TLS elliptic curves extension.
  Only curve names supported by OpenSSL should be used (eg. "P-256","X25519").
  */
@property(nonatomic, readonly) NSArray<NSString *> *tlsEllipticCurves;

- (nonnull instancetype)init NS_UNAVAILABLE;

/** Convenience initializer for a server with no authentication (e.g. STUN). */
- (instancetype)initWithURLStrings:(NSArray<NSString *> *)urlStrings;

/**
 * Initialize an RTCIceServer with its associated URLs, optional username,
 * optional credential, and credentialType.
 */
- (instancetype)initWithURLStrings:(NSArray<NSString *> *)urlStrings
                          username:(nullable NSString *)username
                        credential:(nullable NSString *)credential;

/**
 * Initialize an RTCIceServer with its associated URLs, optional username,
 * optional credential, and TLS cert policy.
 */
- (instancetype)initWithURLStrings:(NSArray<NSString *> *)urlStrings
                          username:(nullable NSString *)username
                        credential:(nullable NSString *)credential
                     tlsCertPolicy:(RTCTlsCertPolicy)tlsCertPolicy;

/**
 * Initialize an RTCIceServer with its associated URLs, optional username,
 * optional credential, TLS cert policy and hostname.
 */
- (instancetype)initWithURLStrings:(NSArray<NSString *> *)urlStrings
                          username:(nullable NSString *)username
                        credential:(nullable NSString *)credential
                     tlsCertPolicy:(RTCTlsCertPolicy)tlsCertPolicy
                          hostname:(nullable NSString *)hostname;

/**
 * Initialize an RTCIceServer with its associated URLs, optional username,
 * optional credential, TLS cert policy, hostname and ALPN protocols.
 */
- (instancetype)initWithURLStrings:(NSArray<NSString *> *)urlStrings
                          username:(nullable NSString *)username
                        credential:(nullable NSString *)credential
                     tlsCertPolicy:(RTCTlsCertPolicy)tlsCertPolicy
                          hostname:(nullable NSString *)hostname
                  tlsAlpnProtocols:(NSArray<NSString *> *)tlsAlpnProtocols;

/**
 * Initialize an RTCIceServer with its associated URLs, optional username,
 * optional credential, TLS cert policy, hostname, ALPN protocols and
 * elliptic curves.
 */
- (instancetype)initWithURLStrings:(NSArray<NSString *> *)urlStrings
                          username:(nullable NSString *)username
                        credential:(nullable NSString *)credential
                     tlsCertPolicy:(RTCTlsCertPolicy)tlsCertPolicy
                          hostname:(nullable NSString *)hostname
                  tlsAlpnProtocols:(nullable NSArray<NSString *> *)tlsAlpnProtocols
                 tlsEllipticCurves:(nullable NSArray<NSString *> *)tlsEllipticCurves
    NS_DESIGNATED_INITIALIZER;

@end

NS_ASSUME_NONNULL_END
