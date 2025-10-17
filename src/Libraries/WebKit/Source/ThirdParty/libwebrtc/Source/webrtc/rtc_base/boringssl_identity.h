/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 12, 2022.
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
#ifndef RTC_BASE_BORINGSSL_IDENTITY_H_
#define RTC_BASE_BORINGSSL_IDENTITY_H_

#include <openssl/ossl_typ.h>

#include <ctime>
#include <memory>
#include <string>

#include "absl/strings/string_view.h"
#include "rtc_base/boringssl_certificate.h"
#include "rtc_base/openssl_key_pair.h"
#include "rtc_base/ssl_certificate.h"
#include "rtc_base/ssl_identity.h"

namespace rtc {

// Holds a keypair and certificate together, and a method to generate them
// consistently. Uses CRYPTO_BUFFER instead of X509, which offers binary size
// and memory improvements.
class BoringSSLIdentity final : public SSLIdentity {
 public:
  static std::unique_ptr<BoringSSLIdentity> CreateWithExpiration(
      absl::string_view common_name,
      const KeyParams& key_params,
      time_t certificate_lifetime);
  static std::unique_ptr<BoringSSLIdentity> CreateForTest(
      const SSLIdentityParams& params);
  static std::unique_ptr<SSLIdentity> CreateFromPEMStrings(
      absl::string_view private_key,
      absl::string_view certificate);
  static std::unique_ptr<SSLIdentity> CreateFromPEMChainStrings(
      absl::string_view private_key,
      absl::string_view certificate_chain);
  ~BoringSSLIdentity() override;

  BoringSSLIdentity(const BoringSSLIdentity&) = delete;
  BoringSSLIdentity& operator=(const BoringSSLIdentity&) = delete;

  const BoringSSLCertificate& certificate() const override;
  const SSLCertChain& cert_chain() const override;

  // Configure an SSL context object to use our key and certificate.
  bool ConfigureIdentity(SSL_CTX* ctx);

  std::string PrivateKeyToPEMString() const override;
  std::string PublicKeyToPEMString() const override;
  bool operator==(const BoringSSLIdentity& other) const;
  bool operator!=(const BoringSSLIdentity& other) const;

 private:
  BoringSSLIdentity(std::unique_ptr<OpenSSLKeyPair> key_pair,
                    std::unique_ptr<BoringSSLCertificate> certificate);
  BoringSSLIdentity(std::unique_ptr<OpenSSLKeyPair> key_pair,
                    std::unique_ptr<SSLCertChain> cert_chain);
  std::unique_ptr<SSLIdentity> CloneInternal() const override;

  static std::unique_ptr<BoringSSLIdentity> CreateInternal(
      const SSLIdentityParams& params);

  std::unique_ptr<OpenSSLKeyPair> key_pair_;
  std::unique_ptr<SSLCertChain> cert_chain_;
};

}  // namespace rtc

#endif  // RTC_BASE_BORINGSSL_IDENTITY_H_
