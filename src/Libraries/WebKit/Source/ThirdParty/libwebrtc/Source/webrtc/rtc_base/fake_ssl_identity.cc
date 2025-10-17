/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 26, 2023.
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
#include "rtc_base/fake_ssl_identity.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/strings/string_view.h"
#include "rtc_base/checks.h"
#include "rtc_base/message_digest.h"

namespace rtc {

FakeSSLCertificate::FakeSSLCertificate(absl::string_view pem_string)
    : pem_string_(pem_string),
      digest_algorithm_(DIGEST_SHA_1),
      expiration_time_(-1) {}

FakeSSLCertificate::FakeSSLCertificate(const FakeSSLCertificate&) = default;

FakeSSLCertificate::~FakeSSLCertificate() = default;

std::unique_ptr<SSLCertificate> FakeSSLCertificate::Clone() const {
  return std::make_unique<FakeSSLCertificate>(*this);
}

std::string FakeSSLCertificate::ToPEMString() const {
  return pem_string_;
}

void FakeSSLCertificate::ToDER(Buffer* der_buffer) const {
  std::string der_string;
  RTC_CHECK(
      SSLIdentity::PemToDer(kPemTypeCertificate, pem_string_, &der_string));
  der_buffer->SetData(der_string.c_str(), der_string.size());
}

int64_t FakeSSLCertificate::CertificateExpirationTime() const {
  return expiration_time_;
}

void FakeSSLCertificate::SetCertificateExpirationTime(int64_t expiration_time) {
  expiration_time_ = expiration_time;
}

void FakeSSLCertificate::set_digest_algorithm(absl::string_view algorithm) {
  digest_algorithm_ = std::string(algorithm);
}

bool FakeSSLCertificate::GetSignatureDigestAlgorithm(
    std::string* algorithm) const {
  *algorithm = digest_algorithm_;
  return true;
}

bool FakeSSLCertificate::ComputeDigest(absl::string_view algorithm,
                                       unsigned char* digest,
                                       size_t size,
                                       size_t* length) const {
  *length = rtc::ComputeDigest(algorithm, pem_string_.c_str(),
                               pem_string_.size(), digest, size);
  return (*length != 0);
}

FakeSSLIdentity::FakeSSLIdentity(absl::string_view pem_string)
    : FakeSSLIdentity(FakeSSLCertificate(pem_string)) {}

FakeSSLIdentity::FakeSSLIdentity(const std::vector<std::string>& pem_strings) {
  std::vector<std::unique_ptr<SSLCertificate>> certs;
  certs.reserve(pem_strings.size());
  for (const std::string& pem_string : pem_strings) {
    certs.push_back(std::make_unique<FakeSSLCertificate>(pem_string));
  }
  cert_chain_ = std::make_unique<SSLCertChain>(std::move(certs));
}

FakeSSLIdentity::FakeSSLIdentity(const FakeSSLCertificate& cert)
    : cert_chain_(std::make_unique<SSLCertChain>(cert.Clone())) {}

FakeSSLIdentity::FakeSSLIdentity(const FakeSSLIdentity& o)
    : cert_chain_(o.cert_chain_->Clone()) {}

FakeSSLIdentity::~FakeSSLIdentity() = default;

std::unique_ptr<SSLIdentity> FakeSSLIdentity::CloneInternal() const {
  return std::make_unique<FakeSSLIdentity>(*this);
}

const SSLCertificate& FakeSSLIdentity::certificate() const {
  return cert_chain_->Get(0);
}

const SSLCertChain& FakeSSLIdentity::cert_chain() const {
  return *cert_chain_.get();
}

std::string FakeSSLIdentity::PrivateKeyToPEMString() const {
  RTC_DCHECK_NOTREACHED();  // Not implemented.
  return "";
}

std::string FakeSSLIdentity::PublicKeyToPEMString() const {
  RTC_DCHECK_NOTREACHED();  // Not implemented.
  return "";
}

bool FakeSSLIdentity::operator==(const SSLIdentity& other) const {
  RTC_DCHECK_NOTREACHED();  // Not implemented.
  return false;
}

}  // namespace rtc
