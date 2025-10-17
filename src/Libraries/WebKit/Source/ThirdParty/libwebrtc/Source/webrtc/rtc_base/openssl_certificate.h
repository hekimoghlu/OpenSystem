/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 15, 2022.
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
#ifndef RTC_BASE_OPENSSL_CERTIFICATE_H_
#define RTC_BASE_OPENSSL_CERTIFICATE_H_

#include <openssl/ossl_typ.h>
#include <stddef.h>
#include <stdint.h>

#include <string>

#include "rtc_base/buffer.h"
#include "rtc_base/ssl_certificate.h"
#include "rtc_base/ssl_identity.h"

namespace rtc {

class OpenSSLKeyPair;

// OpenSSLCertificate encapsulates an OpenSSL X509* certificate object,
// which is also reference counted inside the OpenSSL library.
class OpenSSLCertificate final : public SSLCertificate {
 public:
  // X509 object has its reference count incremented. So the caller and
  // OpenSSLCertificate share ownership.
  explicit OpenSSLCertificate(X509* x509);

  static std::unique_ptr<OpenSSLCertificate> Generate(
      OpenSSLKeyPair* key_pair,
      const SSLIdentityParams& params);
  static std::unique_ptr<OpenSSLCertificate> FromPEMString(
      absl::string_view pem_string);

  ~OpenSSLCertificate() override;

  OpenSSLCertificate(const OpenSSLCertificate&) = delete;
  OpenSSLCertificate& operator=(const OpenSSLCertificate&) = delete;

  std::unique_ptr<SSLCertificate> Clone() const override;

  X509* x509() const { return x509_; }

  std::string ToPEMString() const override;
  void ToDER(Buffer* der_buffer) const override;
  bool operator==(const OpenSSLCertificate& other) const;
  bool operator!=(const OpenSSLCertificate& other) const;

  // Compute the digest of the certificate given algorithm
  bool ComputeDigest(absl::string_view algorithm,
                     unsigned char* digest,
                     size_t size,
                     size_t* length) const override;

  // Compute the digest of a certificate as an X509 *
  static bool ComputeDigest(const X509* x509,
                            absl::string_view algorithm,
                            unsigned char* digest,
                            size_t size,
                            size_t* length);

  bool GetSignatureDigestAlgorithm(std::string* algorithm) const override;

  int64_t CertificateExpirationTime() const override;

 private:
  X509* x509_;  // NOT OWNED
};

}  // namespace rtc

#endif  // RTC_BASE_OPENSSL_CERTIFICATE_H_
