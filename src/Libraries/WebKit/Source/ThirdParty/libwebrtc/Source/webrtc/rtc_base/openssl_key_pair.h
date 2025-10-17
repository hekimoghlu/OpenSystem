/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 13, 2025.
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
#ifndef RTC_BASE_OPENSSL_KEY_PAIR_H_
#define RTC_BASE_OPENSSL_KEY_PAIR_H_

#include <openssl/ossl_typ.h>

#include <memory>
#include <string>

#include "absl/strings/string_view.h"
#include "rtc_base/checks.h"
#include "rtc_base/ssl_identity.h"

namespace rtc {

// OpenSSLKeyPair encapsulates an OpenSSL EVP_PKEY* keypair object,
// which is reference counted inside the OpenSSL library.
class OpenSSLKeyPair final {
 public:
  // Takes ownership of the key.
  explicit OpenSSLKeyPair(EVP_PKEY* pkey) : pkey_(pkey) {
    RTC_DCHECK(pkey_ != nullptr);
  }

  static std::unique_ptr<OpenSSLKeyPair> Generate(const KeyParams& key_params);
  // Constructs a key pair from the private key PEM string. This must not result
  // in missing public key parameters. Returns null on error.
  static std::unique_ptr<OpenSSLKeyPair> FromPrivateKeyPEMString(
      absl::string_view pem_string);

  ~OpenSSLKeyPair();

  OpenSSLKeyPair(const OpenSSLKeyPair&) = delete;
  OpenSSLKeyPair& operator=(const OpenSSLKeyPair&) = delete;

  std::unique_ptr<OpenSSLKeyPair> Clone();

  EVP_PKEY* pkey() const { return pkey_; }
  std::string PrivateKeyToPEMString() const;
  std::string PublicKeyToPEMString() const;
  bool operator==(const OpenSSLKeyPair& other) const;
  bool operator!=(const OpenSSLKeyPair& other) const;

 private:
  void AddReference();

  EVP_PKEY* pkey_;
};

}  // namespace rtc

#endif  // RTC_BASE_OPENSSL_KEY_PAIR_H_
