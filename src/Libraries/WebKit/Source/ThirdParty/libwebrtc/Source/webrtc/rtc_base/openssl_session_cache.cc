/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 29, 2022.
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
#include "rtc_base/openssl_session_cache.h"

#include "absl/strings/string_view.h"
#include "rtc_base/checks.h"
#include "rtc_base/openssl.h"

namespace rtc {

OpenSSLSessionCache::OpenSSLSessionCache(SSLMode ssl_mode, SSL_CTX* ssl_ctx)
    : ssl_mode_(ssl_mode), ssl_ctx_(ssl_ctx) {
  // It is invalid to pass in a null context.
  RTC_DCHECK(ssl_ctx != nullptr);
  SSL_CTX_up_ref(ssl_ctx);
}

OpenSSLSessionCache::~OpenSSLSessionCache() {
  for (const auto& it : sessions_) {
    SSL_SESSION_free(it.second);
  }
  SSL_CTX_free(ssl_ctx_);
}

SSL_SESSION* OpenSSLSessionCache::LookupSession(
    absl::string_view hostname) const {
  auto it = sessions_.find(hostname);
  return (it != sessions_.end()) ? it->second : nullptr;
}

void OpenSSLSessionCache::AddSession(absl::string_view hostname,
                                     SSL_SESSION* new_session) {
  SSL_SESSION* old_session = LookupSession(hostname);
  SSL_SESSION_free(old_session);
  sessions_.insert_or_assign(std::string(hostname), new_session);
}

SSL_CTX* OpenSSLSessionCache::GetSSLContext() const {
  return ssl_ctx_;
}

SSLMode OpenSSLSessionCache::GetSSLMode() const {
  return ssl_mode_;
}

}  // namespace rtc
