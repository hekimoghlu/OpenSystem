/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 28, 2021.
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

#include <wtf/RetainPtr.h>
#include <wtf/Vector.h>
#include <wtf/cf/TypeCastsCF.h>

#if PLATFORM(COCOA)
#include <Security/SecCertificate.h>
#include <Security/SecTrust.h>
#include <wtf/spi/cocoa/SecuritySPI.h>

WTF_DECLARE_CF_TYPE_TRAIT(SecCertificate);
#endif

namespace WebCore {

struct CertificateSummary;

class CertificateInfo {
public:
    CertificateInfo() = default;
    explicit CertificateInfo(RetainPtr<SecTrustRef>&& trust)
        : m_trust(WTFMove(trust))
    {
    }
    const RetainPtr<SecTrustRef>& trust() const { return m_trust; }
    CertificateInfo isolatedCopy() const { return *this; }

    WEBCORE_EXPORT bool containsNonRootSHA1SignedCertificate() const;

    std::optional<CertificateSummary> summary() const;

    bool isEmpty() const
    {
        return !m_trust;
    }

    friend bool operator==(const CertificateInfo&, const CertificateInfo&) = default;

    WEBCORE_EXPORT static RetainPtr<CFArrayRef> certificateChainFromSecTrust(SecTrustRef);
    WEBCORE_EXPORT static RetainPtr<SecTrustRef> secTrustFromCertificateChain(CFArrayRef);

#ifndef NDEBUG
    void dump() const;
#endif

private:
    RetainPtr<SecTrustRef> m_trust;
};

WEBCORE_EXPORT bool certificatesMatch(SecTrustRef, SecTrustRef);

} // namespace WebCore
