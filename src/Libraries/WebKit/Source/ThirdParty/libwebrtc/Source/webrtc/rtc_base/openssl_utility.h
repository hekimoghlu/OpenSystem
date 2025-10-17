/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 2, 2022.
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
#ifndef RTC_BASE_OPENSSL_UTILITY_H_
#define RTC_BASE_OPENSSL_UTILITY_H_

#include <openssl/ossl_typ.h>

#include <string>

#include "absl/strings/string_view.h"

namespace rtc {
// The openssl namespace holds static helper methods. All methods related
// to OpenSSL that are commonly used and don't require global state should be
// placed here.
namespace openssl {

#ifdef OPENSSL_IS_BORINGSSL
// Does minimal parsing of a certificate (only verifying the presence of major
// fields), primarily for the purpose of extracting the relevant out
// parameters. Any that the caller is uninterested in can be null.
bool ParseCertificate(CRYPTO_BUFFER* cert_buffer,
                      CBS* signature_algorithm_oid,
                      int64_t* expiration_time);
#endif

// Verifies that the hostname provided matches that in the peer certificate
// attached to this SSL state.
// TODO(crbug.com/webrtc/11710): When OS certificate verification is available,
// skip compiling this as it adds a dependency on OpenSSL X509 objects, which we
// are trying to avoid in favor of CRYPTO_BUFFERs (see crbug.com/webrtc/11410).
bool VerifyPeerCertMatchesHost(SSL* ssl, absl::string_view host);

// Logs all the errors in the OpenSSL errror queue from the current thread. A
// prefix can be provided for context.
void LogSSLErrors(absl::string_view prefix);

#ifndef WEBRTC_EXCLUDE_BUILT_IN_SSL_ROOT_CERTS
// Attempt to add the certificates from the loader into the SSL_CTX. False is
// returned only if there are no certificates returned from the loader or none
// of them can be added to the TrustStore for the provided context.
bool LoadBuiltinSSLRootCertificates(SSL_CTX* ssl_ctx);
#endif  // WEBRTC_EXCLUDE_BUILT_IN_SSL_ROOT_CERTS

#ifdef OPENSSL_IS_BORINGSSL
CRYPTO_BUFFER_POOL* GetBufferPool();
#endif

}  // namespace openssl
}  // namespace rtc

#endif  // RTC_BASE_OPENSSL_UTILITY_H_
