/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 5, 2023.
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
#ifndef RTC_BASE_TEST_CERTIFICATE_VERIFIER_H_
#define RTC_BASE_TEST_CERTIFICATE_VERIFIER_H_

#include "rtc_base/ssl_certificate.h"

namespace rtc {

class TestCertificateVerifier : public SSLCertificateVerifier {
 public:
  TestCertificateVerifier() = default;
  ~TestCertificateVerifier() override = default;

  bool Verify(const SSLCertificate& certificate) override {
    call_count_++;
    return verify_certificate_;
  }

  size_t call_count_ = 0;
  bool verify_certificate_ = true;
};

}  // namespace rtc

#endif  // RTC_BASE_TEST_CERTIFICATE_VERIFIER_H_
