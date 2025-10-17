/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 17, 2025.
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
#include "rtc_base/rtc_certificate_generator.h"

#include <time.h>

#include <algorithm>
#include <memory>
#include <utility>

#include "rtc_base/checks.h"
#include "rtc_base/ssl_identity.h"

namespace rtc {

namespace {

// A certificates' subject and issuer name.
const char kIdentityName[] = "WebRTC";
const uint64_t kYearInSeconds = 365 * 24 * 60 * 60;

}  // namespace

// static
scoped_refptr<RTCCertificate> RTCCertificateGenerator::GenerateCertificate(
    const KeyParams& key_params,
    const std::optional<uint64_t>& expires_ms) {
  if (!key_params.IsValid()) {
    return nullptr;
  }

  std::unique_ptr<SSLIdentity> identity;
  if (!expires_ms) {
    identity = SSLIdentity::Create(kIdentityName, key_params);
  } else {
    uint64_t expires_s = *expires_ms / 1000;
    // Limit the expiration time to something reasonable (a year). This was
    // somewhat arbitrarily chosen. It also ensures that the value is not too
    // large for the unspecified `time_t`.
    expires_s = std::min(expires_s, kYearInSeconds);
    // TODO(torbjorng): Stop using `time_t`, its type is unspecified. It it safe
    // to assume it can hold up to a year's worth of seconds (and more), but
    // `SSLIdentity::Create` should stop relying on `time_t`.
    // See bugs.webrtc.org/5720.
    time_t cert_lifetime_s = static_cast<time_t>(expires_s);
    identity = SSLIdentity::Create(kIdentityName, key_params, cert_lifetime_s);
  }
  if (!identity) {
    return nullptr;
  }
  return RTCCertificate::Create(std::move(identity));
}

RTCCertificateGenerator::RTCCertificateGenerator(Thread* signaling_thread,
                                                 Thread* worker_thread)
    : signaling_thread_(signaling_thread), worker_thread_(worker_thread) {
  RTC_DCHECK(signaling_thread_);
  RTC_DCHECK(worker_thread_);
}

void RTCCertificateGenerator::GenerateCertificateAsync(
    const KeyParams& key_params,
    const std::optional<uint64_t>& expires_ms,
    RTCCertificateGenerator::Callback callback) {
  RTC_DCHECK(signaling_thread_->IsCurrent());
  RTC_DCHECK(callback);

  worker_thread_->PostTask([key_params, expires_ms,
                            signaling_thread = signaling_thread_,
                            cb = std::move(callback)]() mutable {
    scoped_refptr<RTCCertificate> certificate =
        RTCCertificateGenerator::GenerateCertificate(key_params, expires_ms);
    signaling_thread->PostTask(
        [cert = std::move(certificate), cb = std::move(cb)]() mutable {
          std::move(cb)(std::move(cert));
        });
  });
}

}  // namespace rtc
