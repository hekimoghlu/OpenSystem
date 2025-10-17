/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 17, 2022.
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

#include <memory>

#include "rtc_base/checks.h"
#include "rtc_base/ssl_certificate.h"
#include "rtc_base/ssl_identity.h"
#include "rtc_base/time_utils.h"

namespace rtc {

scoped_refptr<RTCCertificate> RTCCertificate::Create(
    std::unique_ptr<SSLIdentity> identity) {
  // Explicit new to access protected constructor.
  return rtc::scoped_refptr<RTCCertificate>(
      new RTCCertificate(identity.release()));
}

RTCCertificate::RTCCertificate(SSLIdentity* identity) : identity_(identity) {
  RTC_DCHECK(identity_);
}

RTCCertificate::~RTCCertificate() = default;

uint64_t RTCCertificate::Expires() const {
  int64_t expires = GetSSLCertificate().CertificateExpirationTime();
  if (expires != -1)
    return static_cast<uint64_t>(expires) * kNumMillisecsPerSec;
  // If the expiration time could not be retrieved return an expired timestamp.
  return 0;  // = 1970-01-01
}

bool RTCCertificate::HasExpired(uint64_t now) const {
  return Expires() <= now;
}

const SSLCertificate& RTCCertificate::GetSSLCertificate() const {
  return identity_->certificate();
}

const SSLCertChain& RTCCertificate::GetSSLCertificateChain() const {
  return identity_->cert_chain();
}

RTCCertificatePEM RTCCertificate::ToPEM() const {
  return RTCCertificatePEM(identity_->PrivateKeyToPEMString(),
                           GetSSLCertificate().ToPEMString());
}

scoped_refptr<RTCCertificate> RTCCertificate::FromPEM(
    const RTCCertificatePEM& pem) {
  std::unique_ptr<SSLIdentity> identity(
      SSLIdentity::CreateFromPEMStrings(pem.private_key(), pem.certificate()));
  if (!identity)
    return nullptr;
  return RTCCertificate::Create(std::move(identity));
}

bool RTCCertificate::operator==(const RTCCertificate& certificate) const {
  return *this->identity_ == *certificate.identity_;
}

bool RTCCertificate::operator!=(const RTCCertificate& certificate) const {
  return !(*this == certificate);
}

}  // namespace rtc
