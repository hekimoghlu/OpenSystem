/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 10, 2024.
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
#include "rtc_base/openssl_digest.h"

#include "absl/strings/string_view.h"
#include "rtc_base/checks.h"  // RTC_DCHECK, RTC_CHECK
#include "rtc_base/openssl.h"

namespace rtc {

OpenSSLDigest::OpenSSLDigest(absl::string_view algorithm) {
  ctx_ = EVP_MD_CTX_new();
  RTC_CHECK(ctx_ != nullptr);
  EVP_MD_CTX_init(ctx_);
  if (GetDigestEVP(algorithm, &md_)) {
    EVP_DigestInit_ex(ctx_, md_, nullptr);
  } else {
    md_ = nullptr;
  }
}

OpenSSLDigest::~OpenSSLDigest() {
  EVP_MD_CTX_destroy(ctx_);
}

size_t OpenSSLDigest::Size() const {
  if (!md_) {
    return 0;
  }
  return EVP_MD_size(md_);
}

void OpenSSLDigest::Update(const void* buf, size_t len) {
  if (!md_) {
    return;
  }
  EVP_DigestUpdate(ctx_, buf, len);
}

size_t OpenSSLDigest::Finish(void* buf, size_t len) {
  if (!md_ || len < Size()) {
    return 0;
  }
  unsigned int md_len;
  EVP_DigestFinal_ex(ctx_, static_cast<unsigned char*>(buf), &md_len);
  EVP_DigestInit_ex(ctx_, md_, nullptr);  // prepare for future Update()s
  RTC_DCHECK(md_len == Size());
  return md_len;
}

bool OpenSSLDigest::GetDigestEVP(absl::string_view algorithm,
                                 const EVP_MD** mdp) {
  const EVP_MD* md;
  if (algorithm == DIGEST_MD5) {
    md = EVP_md5();
  } else if (algorithm == DIGEST_SHA_1) {
    md = EVP_sha1();
  } else if (algorithm == DIGEST_SHA_224) {
    md = EVP_sha224();
  } else if (algorithm == DIGEST_SHA_256) {
    md = EVP_sha256();
  } else if (algorithm == DIGEST_SHA_384) {
    md = EVP_sha384();
  } else if (algorithm == DIGEST_SHA_512) {
    md = EVP_sha512();
  } else {
    return false;
  }

  // Can't happen
  RTC_DCHECK(EVP_MD_size(md) >= 16);
  *mdp = md;
  return true;
}

bool OpenSSLDigest::GetDigestName(const EVP_MD* md, std::string* algorithm) {
  RTC_DCHECK(md != nullptr);
  RTC_DCHECK(algorithm != nullptr);

  int md_type = EVP_MD_type(md);
  if (md_type == NID_md5) {
    *algorithm = DIGEST_MD5;
  } else if (md_type == NID_sha1) {
    *algorithm = DIGEST_SHA_1;
  } else if (md_type == NID_sha224) {
    *algorithm = DIGEST_SHA_224;
  } else if (md_type == NID_sha256) {
    *algorithm = DIGEST_SHA_256;
  } else if (md_type == NID_sha384) {
    *algorithm = DIGEST_SHA_384;
  } else if (md_type == NID_sha512) {
    *algorithm = DIGEST_SHA_512;
  } else {
    algorithm->clear();
    return false;
  }

  return true;
}

bool OpenSSLDigest::GetDigestSize(absl::string_view algorithm, size_t* length) {
  const EVP_MD* md;
  if (!GetDigestEVP(algorithm, &md))
    return false;

  *length = EVP_MD_size(md);
  return true;
}

}  // namespace rtc
