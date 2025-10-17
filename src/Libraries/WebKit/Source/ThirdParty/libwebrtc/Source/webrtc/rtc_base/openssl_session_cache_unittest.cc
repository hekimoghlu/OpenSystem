/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 29, 2025.
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

#include <openssl/ssl.h>
#include <stdlib.h>

#include <map>
#include <memory>

#include "rtc_base/gunit.h"
#include "rtc_base/openssl.h"

namespace {
// Use methods that avoid X509 objects if possible.
SSL_CTX* NewDtlsContext() {
#ifdef OPENSSL_IS_BORINGSSL
  return SSL_CTX_new(DTLS_with_buffers_method());
#else
  return SSL_CTX_new(DTLS_method());
#endif
}
SSL_CTX* NewTlsContext() {
#ifdef OPENSSL_IS_BORINGSSL
  return SSL_CTX_new(TLS_with_buffers_method());
#else
  return SSL_CTX_new(TLS_method());
#endif
}

SSL_SESSION* NewSslSession(SSL_CTX* ssl_ctx) {
#ifdef OPENSSL_IS_BORINGSSL
  return SSL_SESSION_new(ssl_ctx);
#else
  return SSL_SESSION_new();
#endif
}

}  // namespace

namespace rtc {

TEST(OpenSSLSessionCache, DTLSModeSetCorrectly) {
  SSL_CTX* ssl_ctx = NewDtlsContext();

  OpenSSLSessionCache session_cache(SSL_MODE_DTLS, ssl_ctx);
  EXPECT_EQ(session_cache.GetSSLMode(), SSL_MODE_DTLS);

  SSL_CTX_free(ssl_ctx);
}

TEST(OpenSSLSessionCache, TLSModeSetCorrectly) {
  SSL_CTX* ssl_ctx = NewTlsContext();

  OpenSSLSessionCache session_cache(SSL_MODE_TLS, ssl_ctx);
  EXPECT_EQ(session_cache.GetSSLMode(), SSL_MODE_TLS);

  SSL_CTX_free(ssl_ctx);
}

TEST(OpenSSLSessionCache, SSLContextSetCorrectly) {
  SSL_CTX* ssl_ctx = NewDtlsContext();

  OpenSSLSessionCache session_cache(SSL_MODE_DTLS, ssl_ctx);
  EXPECT_EQ(session_cache.GetSSLContext(), ssl_ctx);

  SSL_CTX_free(ssl_ctx);
}

TEST(OpenSSLSessionCache, InvalidLookupReturnsNullptr) {
  SSL_CTX* ssl_ctx = NewDtlsContext();

  OpenSSLSessionCache session_cache(SSL_MODE_DTLS, ssl_ctx);
  EXPECT_EQ(session_cache.LookupSession("Invalid"), nullptr);
  EXPECT_EQ(session_cache.LookupSession(""), nullptr);
  EXPECT_EQ(session_cache.LookupSession("."), nullptr);

  SSL_CTX_free(ssl_ctx);
}

TEST(OpenSSLSessionCache, SimpleValidSessionLookup) {
  SSL_CTX* ssl_ctx = NewDtlsContext();
  SSL_SESSION* ssl_session = NewSslSession(ssl_ctx);

  OpenSSLSessionCache session_cache(SSL_MODE_DTLS, ssl_ctx);
  session_cache.AddSession("webrtc.org", ssl_session);
  EXPECT_EQ(session_cache.LookupSession("webrtc.org"), ssl_session);

  SSL_CTX_free(ssl_ctx);
}

TEST(OpenSSLSessionCache, AddToExistingReplacesPrevious) {
  SSL_CTX* ssl_ctx = NewDtlsContext();
  SSL_SESSION* ssl_session_1 = NewSslSession(ssl_ctx);
  SSL_SESSION* ssl_session_2 = NewSslSession(ssl_ctx);

  OpenSSLSessionCache session_cache(SSL_MODE_DTLS, ssl_ctx);
  session_cache.AddSession("webrtc.org", ssl_session_1);
  session_cache.AddSession("webrtc.org", ssl_session_2);
  EXPECT_EQ(session_cache.LookupSession("webrtc.org"), ssl_session_2);

  SSL_CTX_free(ssl_ctx);
}

}  // namespace rtc
