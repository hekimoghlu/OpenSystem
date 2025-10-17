/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 3, 2022.
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

NS_ASSUME_NONNULL_BEGIN

/**
 * Objective-C bindings for webrtc::CryptoOptions. This API had to be flattened
 * as Objective-C doesn't support nested structures.
 */
RTC_OBJC_EXPORT
@interface RTCCryptoOptions : NSObject

/**
 * Enable GCM crypto suites from RFC 7714 for SRTP. GCM will only be used
 * if both sides enable it
 */
@property(nonatomic, assign) BOOL srtpEnableGcmCryptoSuites;
/**
 * If set to true, the (potentially insecure) crypto cipher
 * SRTP_AES128_CM_SHA1_32 will be included in the list of supported ciphers
 * during negotiation. It will only be used if both peers support it and no
 * other ciphers get preferred.
 */
@property(nonatomic, assign) BOOL srtpEnableAes128Sha1_32CryptoCipher;
/**
 * If set to true, encrypted RTP header extensions as defined in RFC 6904
 * will be negotiated. They will only be used if both peers support them.
 */
@property(nonatomic, assign) BOOL srtpEnableEncryptedRtpHeaderExtensions;

/**
 * If set all RtpSenders must have an FrameEncryptor attached to them before
 * they are allowed to send packets. All RtpReceivers must have a
 * FrameDecryptor attached to them before they are able to receive packets.
 */
@property(nonatomic, assign) BOOL sframeRequireFrameEncryption;

/**
 * Initializes CryptoOptions with all possible options set explicitly. This
 * is done when converting from a native RTCConfiguration.crypto_options.
 */
- (instancetype)initWithSrtpEnableGcmCryptoSuites:(BOOL)srtpEnableGcmCryptoSuites
              srtpEnableAes128Sha1_32CryptoCipher:(BOOL)srtpEnableAes128Sha1_32CryptoCipher
           srtpEnableEncryptedRtpHeaderExtensions:(BOOL)srtpEnableEncryptedRtpHeaderExtensions
                     sframeRequireFrameEncryption:(BOOL)sframeRequireFrameEncryption
    NS_DESIGNATED_INITIALIZER;

- (instancetype)init NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END
