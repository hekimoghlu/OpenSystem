/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 3, 2024.
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
#include "CertificateInfo.h"

#if USE(CURL)

#include "OpenSSLHelper.h"
#include <openssl/ssl.h>
#include <wtf/CrossThreadCopier.h>

namespace WebCore {

CertificateInfo::CertificateInfo(int verificationError, CertificateChain&& certificateChain)
    : m_verificationError(verificationError)
    , m_certificateChain(WTFMove(certificateChain))
{
}

CertificateInfo CertificateInfo::isolatedCopy() const
{
    return { m_verificationError, crossThreadCopy(m_certificateChain) };
}

String CertificateInfo::verificationErrorDescription() const
{
    return String::fromLatin1(X509_verify_cert_error_string(m_verificationError));
}

CertificateInfo::Certificate CertificateInfo::makeCertificate(std::span<const uint8_t> buffer)
{
    Certificate certificate;
    certificate.append(buffer);
    return certificate;
}

std::optional<CertificateSummary> CertificateInfo::summary() const
{
    if (!m_certificateChain.size())
        return std::nullopt;

    return OpenSSL::createSummaryInfo(m_certificateChain.at(0));
}

}

#endif
