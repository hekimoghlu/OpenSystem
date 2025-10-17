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
#ifndef RTC_BASE_OPENSSL_SESSION_CACHE_H_
#define RTC_BASE_OPENSSL_SESSION_CACHE_H_

#include <openssl/ossl_typ.h>

#include <map>
#include <string>

#include "absl/strings/string_view.h"
#include "rtc_base/ssl_stream_adapter.h"
#include "rtc_base/string_utils.h"

#ifndef OPENSSL_IS_BORINGSSL
typedef struct ssl_session_st SSL_SESSION;
#endif

namespace rtc {

// The OpenSSLSessionCache maps hostnames to SSL_SESSIONS. This cache is
// owned by the OpenSSLAdapterFactory and is passed down to each OpenSSLAdapter
// created with the factory.
class OpenSSLSessionCache final {
 public:
  // Creates a new OpenSSLSessionCache using the provided the SSL_CTX and
  // the ssl_mode. The SSL_CTX will be up_refed. ssl_ctx cannot be nullptr,
  // the constructor immediately dchecks this.
  OpenSSLSessionCache(SSLMode ssl_mode, SSL_CTX* ssl_ctx);
  // Frees the cached SSL_SESSIONS and then frees the SSL_CTX.
  ~OpenSSLSessionCache();

  OpenSSLSessionCache(const OpenSSLSessionCache&) = delete;
  OpenSSLSessionCache& operator=(const OpenSSLSessionCache&) = delete;

  // Looks up a session by hostname. The returned SSL_SESSION is not up_refed.
  SSL_SESSION* LookupSession(absl::string_view hostname) const;
  // Adds a session to the cache, and up_refs it. Any existing session with the
  // same hostname is replaced.
  void AddSession(absl::string_view hostname, SSL_SESSION* session);
  // Returns the true underlying SSL Context that holds these cached sessions.
  SSL_CTX* GetSSLContext() const;
  // The SSL Mode tht the OpenSSLSessionCache was constructed with. This cannot
  // be changed after launch.
  SSLMode GetSSLMode() const;

 private:
  // Holds the SSL Mode that the OpenSSLCache was initialized with. This is
  // immutable after creation and cannot change.
  const SSLMode ssl_mode_;
  /// SSL Context for all shared cached sessions. This SSL_CTX is initialized
  //  with SSL_CTX_set_session_cache_mode(ctx, SSL_SESS_CACHE_CLIENT); Meaning
  //  all client sessions will be added to the cache internal to the context.
  SSL_CTX* ssl_ctx_ = nullptr;
  // Map of hostnames to SSL_SESSIONs; holds references to the SSL_SESSIONs,
  // which are cleaned up when the factory is destroyed.
  // TODO(juberti): Add LRU eviction to keep the cache from growing forever.
  std::map<std::string, SSL_SESSION*, rtc::AbslStringViewCmp> sessions_;
  // The cache should never be copied or assigned directly.
};

}  // namespace rtc

#endif  // RTC_BASE_OPENSSL_SESSION_CACHE_H_
