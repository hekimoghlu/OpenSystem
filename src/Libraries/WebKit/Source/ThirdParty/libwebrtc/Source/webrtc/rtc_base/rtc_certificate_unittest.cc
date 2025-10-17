/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 25, 2024.
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
#include "rtc_base/rtc_certificate.h"

#include <time.h>

#include <memory>
#include <utility>

#include "rtc_base/checks.h"
#include "rtc_base/numerics/safe_conversions.h"
#include "rtc_base/ssl_identity.h"
#include "rtc_base/time_utils.h"
#include "test/gtest.h"

namespace rtc {

namespace {

static const char* kTestCertCommonName = "RTCCertificateTest's certificate";

}  // namespace

class RTCCertificateTest : public ::testing::Test {
 protected:
  scoped_refptr<RTCCertificate> GenerateECDSA() {
    std::unique_ptr<SSLIdentity> identity(
        SSLIdentity::Create(kTestCertCommonName, KeyParams::ECDSA()));
    RTC_CHECK(identity);
    return RTCCertificate::Create(std::move(identity));
  }

  // Timestamp note:
  //   All timestamps in this unittest are expressed in number of seconds since
  // epoch, 1970-01-01T00:00:00Z (UTC). The RTCCertificate interface uses ms,
  // but only seconds-precision is supported by SSLCertificate. To make the
  // tests clearer we convert everything to seconds since the precision matters
  // when generating certificates or comparing timestamps.
  //   As a result, ExpiresSeconds and HasExpiredSeconds are used instead of
  // RTCCertificate::Expires and ::HasExpired for ms -> s conversion.

  uint64_t NowSeconds() const { return TimeNanos() / kNumNanosecsPerSec; }

  uint64_t ExpiresSeconds(const scoped_refptr<RTCCertificate>& cert) const {
    uint64_t exp_ms = cert->Expires();
    uint64_t exp_s = exp_ms / kNumMillisecsPerSec;
    // Make sure this did not result in loss of precision.
    RTC_CHECK_EQ(exp_s * kNumMillisecsPerSec, exp_ms);
    return exp_s;
  }

  bool HasExpiredSeconds(const scoped_refptr<RTCCertificate>& cert,
                         uint64_t now_s) const {
    return cert->HasExpired(now_s * kNumMillisecsPerSec);
  }

  // An RTC_CHECK ensures that `expires_s` this is in valid range of time_t as
  // is required by SSLIdentityParams. On some 32-bit systems time_t is limited
  // to < 2^31. On such systems this will fail for expiration times of year 2038
  // or later.
  scoped_refptr<RTCCertificate> GenerateCertificateWithExpires(
      uint64_t expires_s) const {
    RTC_CHECK(IsValueInRangeForNumericType<time_t>(expires_s));

    SSLIdentityParams params;
    params.common_name = kTestCertCommonName;
    params.not_before = 0;
    params.not_after = static_cast<time_t>(expires_s);
    // Certificate type does not matter for our purposes, using ECDSA because it
    // is fast to generate.
    params.key_params = KeyParams::ECDSA();

    std::unique_ptr<SSLIdentity> identity(SSLIdentity::CreateForTest(params));
    return RTCCertificate::Create(std::move(identity));
  }
};

TEST_F(RTCCertificateTest, NewCertificateNotExpired) {
  // Generate a real certificate without specifying the expiration time.
  // Certificate type doesn't matter, using ECDSA because it's fast to generate.
  scoped_refptr<RTCCertificate> certificate = GenerateECDSA();

  uint64_t now = NowSeconds();
  EXPECT_FALSE(HasExpiredSeconds(certificate, now));
  // Even without specifying the expiration time we would expect it to be valid
  // for at least half an hour.
  EXPECT_FALSE(HasExpiredSeconds(certificate, now + 30 * 60));
}

TEST_F(RTCCertificateTest, UsesExpiresAskedFor) {
  uint64_t now = NowSeconds();
  scoped_refptr<RTCCertificate> certificate =
      GenerateCertificateWithExpires(now);
  EXPECT_EQ(now, ExpiresSeconds(certificate));
}

TEST_F(RTCCertificateTest, ExpiresInOneSecond) {
  // Generate a certificate that expires in 1s.
  uint64_t now = NowSeconds();
  scoped_refptr<RTCCertificate> certificate =
      GenerateCertificateWithExpires(now + 1);
  // Now it should not have expired.
  EXPECT_FALSE(HasExpiredSeconds(certificate, now));
  // In 2s it should have expired.
  EXPECT_TRUE(HasExpiredSeconds(certificate, now + 2));
}

TEST_F(RTCCertificateTest, DifferentCertificatesNotEqual) {
  scoped_refptr<RTCCertificate> a = GenerateECDSA();
  scoped_refptr<RTCCertificate> b = GenerateECDSA();
  EXPECT_TRUE(*a != *b);
}

TEST_F(RTCCertificateTest, CloneWithPEMSerialization) {
  scoped_refptr<RTCCertificate> orig = GenerateECDSA();

  // To PEM.
  RTCCertificatePEM orig_pem = orig->ToPEM();
  // Clone from PEM.
  scoped_refptr<RTCCertificate> clone = RTCCertificate::FromPEM(orig_pem);
  EXPECT_TRUE(clone);
  EXPECT_TRUE(*orig == *clone);
  EXPECT_EQ(orig->Expires(), clone->Expires());
}

TEST_F(RTCCertificateTest, FromPEMWithInvalidPEM) {
  RTCCertificatePEM pem("not a valid PEM", "not a valid PEM");
  scoped_refptr<RTCCertificate> certificate = RTCCertificate::FromPEM(pem);
  EXPECT_FALSE(certificate);
}

}  // namespace rtc
