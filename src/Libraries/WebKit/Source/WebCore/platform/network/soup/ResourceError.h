/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 2, 2024.
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

#include "ResourceErrorBase.h"

#if USE(SOUP)

#include "CertificateInfo.h"
#include <wtf/glib/GRefPtr.h>

typedef struct _GTlsCertificate GTlsCertificate;
typedef struct _SoupMessage SoupMessage;

namespace WebCore {

class ResourceError : public ResourceErrorBase {
public:
    ResourceError(Type type = Type::Null)
        : ResourceErrorBase(type)
    {
    }

    ResourceError(const String& domain, int errorCode, const URL& failingURL, const String& localizedDescription, Type = Type::General, IsSanitized = IsSanitized::No);

    struct IPCData {
        Type type;
        String domain;
        int errorCode;
        URL failingURL;
        String localizedDescription;
        IsSanitized isSanitized;
        CertificateInfo certificateInfo;
    };
    WEBCORE_EXPORT static ResourceError fromIPCData(std::optional<IPCData>&&);
    WEBCORE_EXPORT std::optional<IPCData> ipcData() const;

    static ResourceError httpError(SoupMessage*, GError*);
    static ResourceError transportError(const URL&, int statusCode, const String& reasonPhrase);
    static ResourceError genericGError(const URL&, GError*);
    static ResourceError tlsError(const URL&, unsigned tlsErrors, GTlsCertificate*);
    static ResourceError timeoutError(const URL& failingURL);
    static ResourceError authenticationError(SoupMessage*);

    unsigned tlsErrors() const { return m_tlsErrors; }
    void setTLSErrors(unsigned tlsErrors) { m_tlsErrors = tlsErrors; }
    GTlsCertificate* certificate() const { return m_certificate.get(); }
    void setCertificate(GTlsCertificate* certificate) { m_certificate = certificate; }

    ErrorRecoveryMethod errorRecoveryMethod() const { return ErrorRecoveryMethod::NoRecovery; }

    static bool platformCompare(const ResourceError& a, const ResourceError& b);

private:
    friend class ResourceErrorBase;
    void doPlatformIsolatedCopy(const ResourceError&);

    unsigned m_tlsErrors { 0 };
    GRefPtr<GTlsCertificate> m_certificate;
};

} // namespace WebCore

#endif // USE(SOUP)
