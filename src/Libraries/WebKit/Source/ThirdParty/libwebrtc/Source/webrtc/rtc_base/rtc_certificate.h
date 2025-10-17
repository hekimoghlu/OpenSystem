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
#ifndef RTC_BASE_RTC_CERTIFICATE_H_
#define RTC_BASE_RTC_CERTIFICATE_H_

#include <stdint.h>

#include <memory>
#include <string>

#include "absl/base/attributes.h"
#include "absl/strings/string_view.h"
#include "api/ref_counted_base.h"
#include "api/scoped_refptr.h"
#include "rtc_base/system/rtc_export.h"

namespace rtc {

class SSLCertChain;
class SSLCertificate;
class SSLIdentity;

// This class contains PEM strings of an RTCCertificate's private key and
// certificate and acts as a text representation of RTCCertificate. Certificates
// can be serialized and deserialized to and from this format, which allows for
// cloning and storing of certificates to disk. The PEM format is that of
// `SSLIdentity::PrivateKeyToPEMString` and `SSLCertificate::ToPEMString`, e.g.
// the string representations used by OpenSSL.
class RTCCertificatePEM {
 public:
  RTCCertificatePEM(absl::string_view private_key,
                    absl::string_view certificate)
      : private_key_(private_key), certificate_(certificate) {}

  const std::string& private_key() const { return private_key_; }
  const std::string& certificate() const { return certificate_; }

 private:
  std::string private_key_;
  std::string certificate_;
};

// A thin abstraction layer between "lower level crypto stuff" like
// SSLCertificate and WebRTC usage. Takes ownership of some lower level objects,
// reference counting protects these from premature destruction.
class RTC_EXPORT RTCCertificate final
    : public RefCountedNonVirtual<RTCCertificate> {
 public:
  // Takes ownership of `identity`.
  static scoped_refptr<RTCCertificate> Create(
      std::unique_ptr<SSLIdentity> identity);

  // Returns the expiration time in ms relative to epoch, 1970-01-01T00:00:00Z.
  uint64_t Expires() const;
  // Checks if the certificate has expired, where `now` is expressed in ms
  // relative to epoch, 1970-01-01T00:00:00Z.
  bool HasExpired(uint64_t now) const;

  const SSLCertificate& GetSSLCertificate() const;
  const SSLCertChain& GetSSLCertificateChain() const;

  // TODO(hbos): If possible, remove once RTCCertificate and its
  // GetSSLCertificate() is used in all relevant places. Should not pass around
  // raw SSLIdentity* for the sake of accessing SSLIdentity::certificate().
  // However, some places might need SSLIdentity* for its public/private key...
  SSLIdentity* identity() const { return identity_.get(); }

  // To/from PEM, a text representation of the RTCCertificate.
  RTCCertificatePEM ToPEM() const;
  // Can return nullptr if the certificate is invalid.
  static scoped_refptr<RTCCertificate> FromPEM(const RTCCertificatePEM& pem);
  bool operator==(const RTCCertificate& certificate) const;
  bool operator!=(const RTCCertificate& certificate) const;

 protected:
  explicit RTCCertificate(SSLIdentity* identity);

  friend class RefCountedNonVirtual<RTCCertificate>;
  ~RTCCertificate();

 private:
  // The SSLIdentity is the owner of the SSLCertificate. To protect our
  // GetSSLCertificate() we take ownership of `identity_`.
  const std::unique_ptr<SSLIdentity> identity_;
};

}  // namespace rtc

#endif  // RTC_BASE_RTC_CERTIFICATE_H_
