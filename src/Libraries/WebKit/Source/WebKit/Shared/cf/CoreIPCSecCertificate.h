/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 29, 2025.
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

#if USE(CF)

#import <Security/SecCertificate.h>
#import <wtf/RetainPtr.h>
#import <wtf/cf/VectorCF.h>

namespace WebKit {

class CoreIPCSecCertificate {
public:
    CoreIPCSecCertificate(SecCertificateRef certificate)
        : m_certificateData(dataFromCertificate(certificate))
    {
    }

    CoreIPCSecCertificate(RetainPtr<CFDataRef> data)
        : m_certificateData(data)
    {
    }

    CoreIPCSecCertificate(std::span<const uint8_t> data)
        : m_certificateData(adoptCF(CFDataCreate(kCFAllocatorDefault, data.data(), data.size())))
    {
    }

    RetainPtr<SecCertificateRef> createSecCertificate() const
    {
        auto certificate = adoptCF(SecCertificateCreateWithData(kCFAllocatorDefault, m_certificateData.get()));
        ASSERT(certificate);
        return certificate;
    }

    std::span<const uint8_t> dataReference() const
    {
        RELEASE_ASSERT(m_certificateData);
        return span(m_certificateData.get());
    }

private:
    RetainPtr<CFDataRef> dataFromCertificate(SecCertificateRef certificate) const
    {
        ASSERT(certificate);
        auto data = adoptCF(SecCertificateCopyData(certificate));
        ASSERT(data);
        return data;
    }

    RetainPtr<CFDataRef> m_certificateData;
};

} // namespace WebKit

#endif // USE(CF)
