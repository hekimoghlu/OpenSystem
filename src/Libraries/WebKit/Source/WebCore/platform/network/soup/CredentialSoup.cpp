/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 21, 2024.
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
#include "CredentialSoup.h"

namespace WebCore {

Credential::Credential(const Credential& original, CredentialPersistence persistence)
    : CredentialBase(original, persistence)
    , m_certificate(original.certificate())
{
}

Credential::Credential(GTlsCertificate* certificate, CredentialPersistence persistence)
    : CredentialBase({ }, { }, persistence)
    , m_certificate(certificate)
{
}

bool Credential::isEmpty() const
{
    return !m_certificate && CredentialBase::isEmpty();
}

bool Credential::platformCompare(const Credential& a, const Credential& b)
{
    return a.certificate() == b.certificate();
}

Credential Credential::fromIPCData(IPCData&& ipcData)
{
    return WTF::switchOn(WTFMove(ipcData), [](NonPlatformData&& data) {
        return Credential { data.user, data.password, data.persistence };
    }, [](PlatformData&& data) {
        return Credential { data.certificate.get(), data.persistence };
    });
}

auto Credential::ipcData() const -> IPCData
{
    if (encodingRequiresPlatformData()) {
        return PlatformData {
            m_certificate,
            persistence()
        };
    }
    return nonPlatformData();
}

} // namespace WebCore
