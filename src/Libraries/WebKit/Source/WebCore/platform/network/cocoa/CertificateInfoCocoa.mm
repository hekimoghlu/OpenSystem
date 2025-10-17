/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 17, 2023.
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
#import "config.h"
#import "CertificateInfo.h"

namespace WebCore {

#ifndef NDEBUG
void CertificateInfo::dump() const
{
    if (m_trust) {

        auto chain = adoptCF(SecTrustCopyCertificateChain(trust().get()));
        CFIndex entries = CFArrayGetCount(chain.get());
        NSLog(@"CertificateInfo SecTrust\n");
        NSLog(@"  Entries: %ld\n", entries);
        for (CFIndex i = 0; i < entries; ++i) {
            RetainPtr<CFStringRef> summary = adoptCF(SecCertificateCopySubjectSummary(checked_cf_cast<SecCertificateRef>(CFArrayGetValueAtIndex(chain.get(), i))));
            NSLog(@"  %@", (__bridge NSString *)summary.get());
        }
        return;
    }
    NSLog(@"CertificateInfo (Empty)\n");
}
#endif

} // namespace WebCore
