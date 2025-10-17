/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 28, 2022.
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
#include <stddef.h>
#include <stdint.h>

#include <string>

#include "rtc_base/message_digest.h"
#include "rtc_base/ssl_certificate.h"
#include "rtc_base/string_encode.h"

namespace webrtc {

void FuzzOneInput(const uint8_t* data, size_t size) {
  std::string pem_certificate(reinterpret_cast<const char*>(data), size);

  std::unique_ptr<rtc::SSLCertificate> cert =
      rtc::SSLCertificate::FromPEMString(pem_certificate);

  if (cert == nullptr) {
    return;
  }

  cert->Clone();
  cert->GetStats();
  cert->ToPEMString();
  cert->CertificateExpirationTime();

  std::string algorithm;
  cert->GetSignatureDigestAlgorithm(&algorithm);

  unsigned char digest[rtc::MessageDigest::kMaxSize];
  size_t digest_len;
  cert->ComputeDigest(algorithm, digest, rtc::MessageDigest::kMaxSize,
                      &digest_len);

  rtc::Buffer der_buffer;
  cert->ToDER(&der_buffer);
}

}  // namespace webrtc
