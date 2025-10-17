/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 23, 2022.
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
#ifndef RTC_BASE_OPENSSL_DIGEST_H_
#define RTC_BASE_OPENSSL_DIGEST_H_

#include <openssl/ossl_typ.h>
#include <stddef.h>

#include <string>

#include "absl/strings/string_view.h"
#include "rtc_base/message_digest.h"

namespace rtc {

// An implementation of the digest class that uses OpenSSL.
class OpenSSLDigest final : public MessageDigest {
 public:
  // Creates an OpenSSLDigest with `algorithm` as the hash algorithm.
  explicit OpenSSLDigest(absl::string_view algorithm);
  ~OpenSSLDigest() override;
  // Returns the digest output size (e.g. 16 bytes for MD5).
  size_t Size() const override;
  // Updates the digest with `len` bytes from `buf`.
  void Update(const void* buf, size_t len) override;
  // Outputs the digest value to `buf` with length `len`.
  size_t Finish(void* buf, size_t len) override;

  // Helper function to look up a digest's EVP by name.
  static bool GetDigestEVP(absl::string_view algorithm, const EVP_MD** md);
  // Helper function to look up a digest's name by EVP.
  static bool GetDigestName(const EVP_MD* md, std::string* algorithm);
  // Helper function to get the length of a digest.
  static bool GetDigestSize(absl::string_view algorithm, size_t* len);

 private:
  EVP_MD_CTX* ctx_ = nullptr;
  const EVP_MD* md_;
};

}  // namespace rtc

#endif  // RTC_BASE_OPENSSL_DIGEST_H_
