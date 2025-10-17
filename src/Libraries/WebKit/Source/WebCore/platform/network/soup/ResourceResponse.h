/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 6, 2024.
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

#include "ResourceResponseBase.h"

#include <libsoup/soup.h>
#include <wtf/glib/GRefPtr.h>
#include <wtf/text/CString.h>

namespace WebCore {

class ResourceResponse : public ResourceResponseBase {
public:
    ResourceResponse() = default;

    ResourceResponse(const URL& url, const String& mimeType, long long expectedLength, const String& textEncodingName)
        : ResourceResponseBase(url, mimeType, expectedLength, textEncodingName)
    {
    }

    ResourceResponse(ResourceResponseBase&& base)
        : ResourceResponseBase(WTFMove(base))
    {
    }

    ResourceResponse(SoupMessage*, const CString& sniffedContentType = CString());

    void updateSoupMessageHeaders(SoupMessageHeaders*) const;
    void updateFromSoupMessageHeaders(SoupMessageHeaders*);

    GTlsCertificate* soupMessageCertificate() const { return m_certificate.get(); }
    GTlsCertificateFlags soupMessageTLSErrors() const { return m_tlsErrors; }

private:
    friend class ResourceResponseBase;

    String m_sniffedContentType;
    GRefPtr<GTlsCertificate> m_certificate;
    GTlsCertificateFlags m_tlsErrors { static_cast<GTlsCertificateFlags>(0) };

    void doUpdateResourceResponse() { }
    String platformSuggestedFilename() const;
    CertificateInfo platformCertificateInfo(std::span<const std::byte>) const;
};

} // namespace WebCore
