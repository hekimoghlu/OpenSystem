/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 24, 2024.
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
#ifndef RTC_BASE_BORINGSSL_CERTIFICATE_H_
#define RTC_BASE_BORINGSSL_CERTIFICATE_H_

#include <openssl/base.h>
#include <openssl/ossl_typ.h>
#include <stddef.h>
#include <stdint.h>

#include <memory>
#include <string>

#include "absl/strings/string_view.h"
#include "rtc_base/buffer.h"
#include "rtc_base/ssl_certificate.h"
#include "rtc_base/ssl_identity.h"

namespace rtc {

class OpenSSLKeyPair;

// BoringSSLCertificate encapsulates a BoringSSL CRYPTO_BUFFER object holding a
// certificate, which is also reference counted inside the BoringSSL library.
// This offers binary size and memory improvements over the OpenSSL X509
// object.
class BoringSSLCertificate final : public SSLCertificate {
 public:
  explicit BoringSSLCertificate(bssl::UniquePtr<CRYPTO_BUFFER> cert_buffer);

  static std::unique_ptr<BoringSSLCertificate> Generate(
      OpenSSLKeyPair* key_pair,
      const SSLIdentityParams& params);
  static std::unique_ptr<BoringSSLCertificate> FromPEMString(
      absl::string_view pem_string);

  ~BoringSSLCertificate() override;

  BoringSSLCertificate(const BoringSSLCertificate&) = delete;
  BoringSSLCertificate& operator=(const BoringSSLCertificate&) = delete;

  std::unique_ptr<SSLCertificate> Clone() const override;

  CRYPTO_BUFFER* cert_buffer() const { return cert_buffer_.get(); }

  std::string ToPEMString() const override;
  void ToDER(Buffer* der_buffer) const override;
  bool operator==(const BoringSSLCertificate& other) const;
  bool operator!=(const BoringSSLCertificate& other) const;

  // Compute the digest of the certificate given `algorithm`.
  bool ComputeDigest(absl::string_view algorithm,
                     unsigned char* digest,
                     size_t size,
                     size_t* length) const override;

  // Compute the digest of a certificate as a CRYPTO_BUFFER.
  static bool ComputeDigest(const CRYPTO_BUFFER* cert_buffer,
                            absl::string_view algorithm,
                            unsigned char* digest,
                            size_t size,
                            size_t* length);

  bool GetSignatureDigestAlgorithm(std::string* algorithm) const override;

  int64_t CertificateExpirationTime() const override;

 private:
  // A handle to the DER encoded certificate data.
  bssl::UniquePtr<CRYPTO_BUFFER> cert_buffer_;
};

}  // namespace rtc

#endif  // RTC_BASE_BORINGSSL_CERTIFICATE_H_
