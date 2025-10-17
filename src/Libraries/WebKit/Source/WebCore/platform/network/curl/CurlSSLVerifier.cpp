/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 11, 2022.
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
#include "config.h"
#include "CurlSSLVerifier.h"
#include <wtf/TZoneMallocInlines.h>

#if USE(CURL)
#include "CurlContext.h"
#include "CurlSSLHandle.h"

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(x);

CurlSSLVerifier::CurlSSLVerifier(void* sslCtx)
{
    auto* ctx = static_cast<SSL_CTX*>(sslCtx);

    SSL_CTX_set_app_data(ctx, this);
    SSL_CTX_set_verify(ctx, SSL_CTX_get_verify_mode(ctx), verifyCallback);

#if !defined(LIBRESSL_VERSION_NUMBER)
    const auto& sslHandle = CurlContext::singleton().sslHandle();
    if (const auto& signatureAlgorithmsList = sslHandle.signatureAlgorithmsList(); !signatureAlgorithmsList.isNull())
        SSL_CTX_set1_sigalgs_list(ctx, signatureAlgorithmsList.data());
#endif
}

std::unique_ptr<WebCore::CertificateInfo> CurlSSLVerifier::createCertificateInfo(std::optional<long>&& verifyResult)
{
    if (!verifyResult)
        return nullptr;

    if (m_certificateChain.isEmpty())
        return nullptr;

    return makeUnique<CertificateInfo>(*verifyResult, WTFMove(m_certificateChain));
}

void CurlSSLVerifier::collectInfo(X509_STORE_CTX* ctx)
{
    if (!ctx)
        return;

    m_certificateChain = OpenSSL::createCertificateChain(ctx);
}

int CurlSSLVerifier::verifyCallback(int preverified, X509_STORE_CTX* ctx)
{
    auto ssl = static_cast<SSL*>(X509_STORE_CTX_get_ex_data(ctx, SSL_get_ex_data_X509_STORE_CTX_idx()));
    auto sslCtx = SSL_get_SSL_CTX(ssl);
    auto verifier = static_cast<CurlSSLVerifier*>(SSL_CTX_get_app_data(sslCtx));

    verifier->collectInfo(ctx);
    // whether the verification of the certificate in question was passed (preverified=1) or not (preverified=0)
    return preverified;
}

}
#endif
