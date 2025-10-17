/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 8, 2025.
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

#include "CertificateSummary.h"
#include <wtf/persistence/PersistentDecoder.h>
#include <wtf/persistence/PersistentEncoder.h>

namespace WebCore {

bool certificatesMatch(SecTrustRef trust1, SecTrustRef trust2)
{
    if (!trust1 || !trust2)
        return false;

    RetainPtr chain1 = adoptCF(SecTrustCopyCertificateChain(trust1));
    RetainPtr chain2 = adoptCF(SecTrustCopyCertificateChain(trust2));
    CFIndex count1 = chain1 ? CFArrayGetCount(chain1.get()) : 0;
    CFIndex count2 = chain2 ? CFArrayGetCount(chain2.get()) : 0;

    if (count1 != count2)
        return false;

    for (CFIndex i = 0; i < count1; ++i) {
        RetainPtr cert1 = CFArrayGetValueAtIndex(chain1.get(), i);
        RetainPtr cert2 = CFArrayGetValueAtIndex(chain2.get(), i);
        RELEASE_ASSERT(cert1);
        RELEASE_ASSERT(cert2);
        if (!CFEqual(cert1.get(), cert2.get()))
            return false;
    }

    return true;
}

RetainPtr<SecTrustRef> CertificateInfo::secTrustFromCertificateChain(CFArrayRef certificateChain)
{
    SecTrustRef trustRef = nullptr;
    if (SecTrustCreateWithCertificates(certificateChain, nullptr, &trustRef) != noErr)
        return nullptr;
    return adoptCF(trustRef);
}

RetainPtr<CFArrayRef> CertificateInfo::certificateChainFromSecTrust(SecTrustRef trust)
{
    return adoptCF(SecTrustCopyCertificateChain(trust));
}

bool CertificateInfo::containsNonRootSHA1SignedCertificate() const
{
    if (m_trust) {
        auto chain = adoptCF(SecTrustCopyCertificateChain(trust().get()));
        // Allow only the root certificate (the last in the chain) to be SHA1.
        for (CFIndex i = 0, size = SecTrustGetCertificateCount(trust().get()) - 1; i < size; ++i) {
            auto certificate = checked_cf_cast<SecCertificateRef>(CFArrayGetValueAtIndex(chain.get(), i));
            if (SecCertificateGetSignatureHashAlgorithm(certificate) == kSecSignatureHashAlgorithmSHA1)
                return true;
        }

        return false;
    }

    return false;
}

std::optional<CertificateSummary> CertificateInfo::summary() const
{
    CertificateSummary summaryInfo;
    auto chain = certificateChainFromSecTrust(m_trust.get());
    if (!chain || !CFArrayGetCount(chain.get()))
        return std::nullopt;

#if !PLATFORM(IOS_FAMILY_SIMULATOR) && !PLATFORM(MACCATALYST)
    auto leafCertificate = checked_cf_cast<SecCertificateRef>(CFArrayGetValueAtIndex(chain.get(), 0));
    auto subjectCF = adoptCF(SecCertificateCopySubjectSummary(leafCertificate));
    summaryInfo.subject = subjectCF.get();
#endif

#if PLATFORM(MAC)
    if (auto certificateDictionary = adoptCF(SecCertificateCopyValues(leafCertificate, nullptr, nullptr))) {
        // CFAbsoluteTime is relative to 01/01/1970 00:00:00 GMT.
        const Seconds absoluteReferenceDate(978307200);

        if (auto validNotBefore = checked_cf_cast<CFDictionaryRef>(CFDictionaryGetValue(certificateDictionary.get(), kSecOIDX509V1ValidityNotBefore))) {
            if (auto number = checked_cf_cast<CFNumberRef>(CFDictionaryGetValue(validNotBefore, CFSTR("value")))) {
                double numberValue;
                if (CFNumberGetValue(number, kCFNumberDoubleType, &numberValue))
                    summaryInfo.validFrom = absoluteReferenceDate + Seconds(numberValue);
            }
        }

        if (auto validNotAfter = checked_cf_cast<CFDictionaryRef>(CFDictionaryGetValue(certificateDictionary.get(), kSecOIDX509V1ValidityNotAfter))) {
            if (auto number = checked_cf_cast<CFNumberRef>(CFDictionaryGetValue(validNotAfter, CFSTR("value")))) {
                double numberValue;
                if (CFNumberGetValue(number, kCFNumberDoubleType, &numberValue))
                    summaryInfo.validUntil = absoluteReferenceDate + Seconds(numberValue);
            }
        }

        if (auto dnsNames = checked_cf_cast<CFDictionaryRef>(CFDictionaryGetValue(certificateDictionary.get(), CFSTR("DNSNAMES")))) {
            if (auto dnsNamesArray = checked_cf_cast<CFArrayRef>(CFDictionaryGetValue(dnsNames, CFSTR("value")))) {
                for (CFIndex i = 0, count = CFArrayGetCount(dnsNamesArray); i < count; ++i) {
                    if (auto dnsName = checked_cf_cast<CFStringRef>(CFArrayGetValueAtIndex(dnsNamesArray, i)))
                        summaryInfo.dnsNames.append(dnsName);
                }
            }
        }

        if (auto ipAddresses = checked_cf_cast<CFDictionaryRef>(CFDictionaryGetValue(certificateDictionary.get(), CFSTR("IPADDRESSES")))) {
            if (auto ipAddressesArray = checked_cf_cast<CFArrayRef>(CFDictionaryGetValue(ipAddresses, CFSTR("value")))) {
                for (CFIndex i = 0, count = CFArrayGetCount(ipAddressesArray); i < count; ++i) {
                    if (auto ipAddress = checked_cf_cast<CFStringRef>(CFArrayGetValueAtIndex(ipAddressesArray, i)))
                        summaryInfo.ipAddresses.append(ipAddress);
                }
            }
        }
    }
#endif // PLATFORM(MAC)
    return summaryInfo;
}

} // namespace WTF::Persistence
