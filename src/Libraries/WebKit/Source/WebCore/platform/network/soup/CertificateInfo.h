/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 11, 2023.
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
#pragma once

#include "CertificateSummary.h"
#include "NotImplemented.h"
#include <libsoup/soup.h>
#include <wtf/Vector.h>
#include <wtf/glib/GRefPtr.h>
#include <wtf/persistence/PersistentCoders.h>
#include <wtf/persistence/PersistentDecoder.h>
#include <wtf/persistence/PersistentEncoder.h>

namespace WebCore {

class ResourceError;
class ResourceResponse;

class CertificateInfo {
public:
    CertificateInfo();
    explicit CertificateInfo(const WebCore::ResourceResponse&);
    explicit CertificateInfo(const WebCore::ResourceError&);
    CertificateInfo(GRefPtr<GTlsCertificate>&&, GTlsCertificateFlags);
    WEBCORE_EXPORT ~CertificateInfo();

    CertificateInfo isolatedCopy() const;

    const GRefPtr<GTlsCertificate>& certificate() const { return m_certificate; }
    void setCertificate(GTlsCertificate* certificate) { m_certificate = certificate; }
    GTlsCertificateFlags tlsErrors() const { return m_tlsErrors; }
    void setTLSErrors(GTlsCertificateFlags tlsErrors) { m_tlsErrors = tlsErrors; }

    bool containsNonRootSHA1SignedCertificate() const { notImplemented(); return false; }

    std::optional<CertificateSummary> summary() const;

    bool isEmpty() const { return !m_certificate; }

    bool operator==(const CertificateInfo& other) const
    {
        if (tlsErrors() != other.tlsErrors())
            return false;

        if (m_certificate.get() == other.m_certificate.get())
            return true;

        return m_certificate && other.m_certificate && g_tls_certificate_is_same(m_certificate.get(), other.m_certificate.get());
    }

private:
    GRefPtr<GTlsCertificate> m_certificate;
    GTlsCertificateFlags m_tlsErrors;
};

} // namespace WebCore
