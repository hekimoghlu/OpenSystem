/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 13, 2021.
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
#ifndef RTC_BASE_SSL_FINGERPRINT_H_
#define RTC_BASE_SSL_FINGERPRINT_H_

#include <stddef.h>
#include <stdint.h>

#include <string>

#include "absl/strings/string_view.h"
#include "rtc_base/copy_on_write_buffer.h"
#include "rtc_base/system/rtc_export.h"

namespace rtc {

class RTCCertificate;
class SSLCertificate;
class SSLIdentity;

struct RTC_EXPORT SSLFingerprint {
  // TODO(steveanton): Remove once downstream projects have moved off of this.
  static SSLFingerprint* Create(absl::string_view algorithm,
                                const rtc::SSLIdentity* identity);
  // TODO(steveanton): Rename to Create once projects have migrated.
  static std::unique_ptr<SSLFingerprint> CreateUnique(
      absl::string_view algorithm,
      const rtc::SSLIdentity& identity);

  static std::unique_ptr<SSLFingerprint> Create(
      absl::string_view algorithm,
      const rtc::SSLCertificate& cert);

  // TODO(steveanton): Remove once downstream projects have moved off of this.
  static SSLFingerprint* CreateFromRfc4572(absl::string_view algorithm,
                                           absl::string_view fingerprint);
  // TODO(steveanton): Rename to CreateFromRfc4572 once projects have migrated.
  static std::unique_ptr<SSLFingerprint> CreateUniqueFromRfc4572(
      absl::string_view algorithm,
      absl::string_view fingerprint);

  // Creates a fingerprint from a certificate, using the same digest algorithm
  // as the certificate's signature.
  static std::unique_ptr<SSLFingerprint> CreateFromCertificate(
      const RTCCertificate& cert);

  SSLFingerprint(absl::string_view algorithm,
                 ArrayView<const uint8_t> digest_view);
  // TODO(steveanton): Remove once downstream projects have moved off of this.
  SSLFingerprint(absl::string_view algorithm,
                 const uint8_t* digest_in,
                 size_t digest_len);

  SSLFingerprint(const SSLFingerprint& from) = default;
  SSLFingerprint& operator=(const SSLFingerprint& from) = default;

  bool operator==(const SSLFingerprint& other) const;

  std::string GetRfc4572Fingerprint() const;

  std::string ToString() const;

  std::string algorithm;
  rtc::CopyOnWriteBuffer digest;
};

}  // namespace rtc

#endif  // RTC_BASE_SSL_FINGERPRINT_H_
