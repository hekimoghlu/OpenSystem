/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 17, 2024.
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

#if USE(SOUP)

#include "CertificateInfo.h"

#include <ResourceError.h>
#include <ResourceResponse.h>
#include <libsoup/soup.h>
#include <wtf/glib/GSpanExtras.h>

namespace WebCore {

CertificateInfo::CertificateInfo()
    : m_tlsErrors(static_cast<GTlsCertificateFlags>(0))
{
}

CertificateInfo::CertificateInfo(const ResourceResponse& response)
    : m_certificate(response.soupMessageCertificate())
    , m_tlsErrors(response.soupMessageTLSErrors())
{
}

CertificateInfo::CertificateInfo(const ResourceError& resourceError)
    : m_certificate(resourceError.certificate())
    , m_tlsErrors(static_cast<GTlsCertificateFlags>(resourceError.tlsErrors()))
{
}

CertificateInfo::CertificateInfo(GRefPtr<GTlsCertificate>&& certificate, GTlsCertificateFlags tlsErrors)
    : m_certificate(WTFMove(certificate))
    , m_tlsErrors(tlsErrors)
{
}

CertificateInfo::~CertificateInfo() = default;

CertificateInfo CertificateInfo::isolatedCopy() const
{
    if (!m_certificate)
        return { };

    Vector<GUniquePtr<char>> certificatesDataList;
    for (auto* nextCertificate = m_certificate.get(); nextCertificate; nextCertificate = g_tls_certificate_get_issuer(nextCertificate)) {
        GUniqueOutPtr<char> certificateData;
        g_object_get(nextCertificate, "certificate-pem", &certificateData.outPtr(), nullptr);
        certificatesDataList.append(certificateData.release());
    }

    GUniqueOutPtr<char> privateKey;
    GUniqueOutPtr<char> privateKeyPKCS11Uri;
    g_object_get(m_certificate.get(), "private-key-pem", &privateKey.outPtr(), "private-key-pkcs11-uri", &privateKeyPKCS11Uri.outPtr(), nullptr);

    GType certificateType = g_tls_backend_get_certificate_type(g_tls_backend_get_default());
    GRefPtr<GTlsCertificate> certificate;
    GTlsCertificate* issuer = nullptr;
    while (!certificatesDataList.isEmpty()) {
        auto certificateData = certificatesDataList.takeLast();
        certificate = adoptGRef(G_TLS_CERTIFICATE(g_initable_new(
            certificateType, nullptr, nullptr,
            "certificate-pem", certificateData.get(),
            "issuer", issuer,
            "private-key-pem", certificatesDataList.isEmpty() ? privateKey.get() : nullptr,
            "private-key-pkcs11-uri", certificatesDataList.isEmpty() ? privateKeyPKCS11Uri.get() : nullptr,
            nullptr)));
        RELEASE_ASSERT(certificate);
        issuer = certificate.get();
    }

    return CertificateInfo(certificate.get(), m_tlsErrors);
}

std::optional<CertificateSummary> CertificateInfo::summary() const
{
    if (!m_certificate)
        return std::nullopt;

    CertificateSummary summaryInfo;

    GRefPtr<GDateTime> validNotBefore;
    GRefPtr<GDateTime> validNotAfter;
    GUniqueOutPtr<char> subjectName;
    GRefPtr<GPtrArray> dnsNames;
    GRefPtr<GPtrArray> ipAddresses;
    g_object_get(m_certificate.get(), "not-valid-before", &validNotBefore.outPtr(), "not-valid-after", &validNotAfter.outPtr(),
        "subject-name", &subjectName.outPtr(), "dns-names", &dnsNames.outPtr(), "ip-addresses", &ipAddresses.outPtr(), nullptr);

    if (validNotBefore)
        summaryInfo.validFrom = Seconds(static_cast<double>(g_date_time_to_unix(validNotBefore.get())));
    if (validNotAfter)
        summaryInfo.validUntil = Seconds(static_cast<double>(g_date_time_to_unix(validNotAfter.get())));
    if (subjectName)
        summaryInfo.subject = String::fromUTF8(subjectName.get());
    for (auto dnsName : span<GBytes*>(dnsNames))
        summaryInfo.dnsNames.append(String(span(dnsName)));
    for (auto address : span<GInetAddress*>(ipAddresses)) {
        GUniquePtr<char> ipAddress(g_inet_address_to_string(address));
        summaryInfo.ipAddresses.append(String::fromUTF8(ipAddress.get()));
    }

    return summaryInfo;
}

} // namespace WebCore

#endif
