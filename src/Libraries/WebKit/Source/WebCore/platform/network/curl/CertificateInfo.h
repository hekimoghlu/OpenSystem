/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 14, 2025.
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
#include <wtf/TZoneMallocInlines.h>
#include <wtf/Vector.h>
#include <wtf/persistence/PersistentCoders.h>
#include <wtf/persistence/PersistentDecoder.h>
#include <wtf/persistence/PersistentEncoder.h>

namespace WebCore {

class CertificateInfo {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(CertificateInfo);
public:
    using Certificate = Vector<uint8_t>;
    using CertificateChain = Vector<Certificate>;

    CertificateInfo() = default;
    WEBCORE_EXPORT CertificateInfo(int verificationError, CertificateChain&&);

    WEBCORE_EXPORT CertificateInfo isolatedCopy() const;

    int verificationError() const { return m_verificationError; }
    WEBCORE_EXPORT String verificationErrorDescription() const;
    const Vector<Certificate>& certificateChain() const { return m_certificateChain; }

    bool containsNonRootSHA1SignedCertificate() const { notImplemented(); return false; }

    std::optional<CertificateSummary> summary() const;

    bool isEmpty() const { return m_certificateChain.isEmpty(); }

    static Certificate makeCertificate(std::span<const uint8_t>);

    friend bool operator==(const CertificateInfo&, const CertificateInfo&) = default;

private:
    int m_verificationError { 0 };
    CertificateChain m_certificateChain;
};

} // namespace WebCore
