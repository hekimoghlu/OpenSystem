/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 8, 2023.
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
#ifndef API_CRYPTO_CRYPTO_OPTIONS_H_
#define API_CRYPTO_CRYPTO_OPTIONS_H_

#include <vector>

#include "rtc_base/system/rtc_export.h"

namespace webrtc {

// CryptoOptions defines advanced cryptographic settings for native WebRTC.
// These settings must be passed into PeerConnectionFactoryInterface::Options
// and are only applicable to native use cases of WebRTC.
struct RTC_EXPORT CryptoOptions {
  CryptoOptions();
  CryptoOptions(const CryptoOptions& other);
  ~CryptoOptions();

  // Helper method to return an instance of the CryptoOptions with GCM crypto
  // suites disabled. This method should be used instead of depending on current
  // default values set by the constructor.
  static CryptoOptions NoGcm();

  // Returns a list of the supported DTLS-SRTP Crypto suites based on this set
  // of crypto options.
  std::vector<int> GetSupportedDtlsSrtpCryptoSuites() const;

  bool operator==(const CryptoOptions& other) const;
  bool operator!=(const CryptoOptions& other) const;

  // SRTP Related Peer Connection options.
  struct Srtp {
    // Enable GCM crypto suites from RFC 7714 for SRTP. GCM will only be used
    // if both sides enable it.
    bool enable_gcm_crypto_suites = true;

    // If set to true, the (potentially insecure) crypto cipher
    // kSrtpAes128CmSha1_32 will be included in the list of supported ciphers
    // during negotiation. It will only be used if both peers support it and no
    // other ciphers get preferred.
    bool enable_aes128_sha1_32_crypto_cipher = false;

    // The most commonly used cipher. Can be disabled, mostly for testing
    // purposes.
    bool enable_aes128_sha1_80_crypto_cipher = true;

    // If set to true, encrypted RTP header extensions as defined in RFC 6904
    // will be negotiated. They will only be used if both peers support them.
    bool enable_encrypted_rtp_header_extensions = false;
  } srtp;

  // Options to be used when the FrameEncryptor / FrameDecryptor APIs are used.
  struct SFrame {
    // If set all RtpSenders must have an FrameEncryptor attached to them before
    // they are allowed to send packets. All RtpReceivers must have a
    // FrameDecryptor attached to them before they are able to receive packets.
    bool require_frame_encryption = false;
  } sframe;
};

}  // namespace webrtc

#endif  // API_CRYPTO_CRYPTO_OPTIONS_H_
