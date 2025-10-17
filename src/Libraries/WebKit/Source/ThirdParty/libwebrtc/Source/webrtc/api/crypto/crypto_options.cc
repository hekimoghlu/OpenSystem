/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 10, 2024.
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
#include "api/crypto/crypto_options.h"

#include <vector>

#include "rtc_base/checks.h"
#include "rtc_base/ssl_stream_adapter.h"

namespace webrtc {

CryptoOptions::CryptoOptions() {}

CryptoOptions::CryptoOptions(const CryptoOptions& other) {
  srtp = other.srtp;
  sframe = other.sframe;
}

CryptoOptions::~CryptoOptions() {}

// static
CryptoOptions CryptoOptions::NoGcm() {
  CryptoOptions options;
  options.srtp.enable_gcm_crypto_suites = false;
  return options;
}

std::vector<int> CryptoOptions::GetSupportedDtlsSrtpCryptoSuites() const {
  std::vector<int> crypto_suites;
  // Note: kSrtpAes128CmSha1_80 is what is required to be supported (by
  // draft-ietf-rtcweb-security-arch), but kSrtpAes128CmSha1_32 is allowed as
  // well, and saves a few bytes per packet if it ends up selected.
  // As the cipher suite is potentially insecure, it will only be used if
  // enabled by both peers.
  if (srtp.enable_aes128_sha1_32_crypto_cipher) {
    crypto_suites.push_back(rtc::kSrtpAes128CmSha1_32);
  }
  if (srtp.enable_aes128_sha1_80_crypto_cipher) {
    crypto_suites.push_back(rtc::kSrtpAes128CmSha1_80);
  }

  // Note: GCM cipher suites are not the top choice since they increase the
  // packet size. In order to negotiate them the other side must not support
  // kSrtpAes128CmSha1_80.
  if (srtp.enable_gcm_crypto_suites) {
    crypto_suites.push_back(rtc::kSrtpAeadAes256Gcm);
    crypto_suites.push_back(rtc::kSrtpAeadAes128Gcm);
  }
  RTC_CHECK(!crypto_suites.empty());
  return crypto_suites;
}

bool CryptoOptions::operator==(const CryptoOptions& other) const {
  struct data_being_tested_for_equality {
    struct Srtp {
      bool enable_gcm_crypto_suites;
      bool enable_aes128_sha1_32_crypto_cipher;
      bool enable_aes128_sha1_80_crypto_cipher;
      bool enable_encrypted_rtp_header_extensions;
    } srtp;
    struct SFrame {
      bool require_frame_encryption;
    } sframe;
  };
  static_assert(sizeof(data_being_tested_for_equality) == sizeof(*this),
                "Did you add something to CryptoOptions and forget to "
                "update operator==?");

  return srtp.enable_gcm_crypto_suites == other.srtp.enable_gcm_crypto_suites &&
         srtp.enable_aes128_sha1_32_crypto_cipher ==
             other.srtp.enable_aes128_sha1_32_crypto_cipher &&
         srtp.enable_aes128_sha1_80_crypto_cipher ==
             other.srtp.enable_aes128_sha1_80_crypto_cipher &&
         srtp.enable_encrypted_rtp_header_extensions ==
             other.srtp.enable_encrypted_rtp_header_extensions &&
         sframe.require_frame_encryption ==
             other.sframe.require_frame_encryption;
}

bool CryptoOptions::operator!=(const CryptoOptions& other) const {
  return !(*this == other);
}

}  // namespace webrtc
