/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 4, 2025.
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

#include "CredentialBase.h"
#include <wtf/glib/GRefPtr.h>

typedef struct _GTlsCertificate GTlsCertificate;

namespace WebCore {

class Credential : public CredentialBase {
public:
    Credential()
        : CredentialBase()
    {
    }

    Credential(const String& user, const String& password, CredentialPersistence persistence)
        : CredentialBase(user, password, persistence)
    {
    }

    Credential(const Credential&, CredentialPersistence);

    WEBCORE_EXPORT Credential(GTlsCertificate*, CredentialPersistence);

    WEBCORE_EXPORT bool isEmpty() const;

    static bool platformCompare(const Credential&, const Credential&);

    bool encodingRequiresPlatformData() const { return !!m_certificate; }

    GTlsCertificate* certificate() const { return m_certificate.get(); }

    struct PlatformData {
        GRefPtr<GTlsCertificate> certificate;
        CredentialPersistence persistence;
    };

    using IPCData = std::variant<NonPlatformData, PlatformData>;
    WEBCORE_EXPORT static Credential fromIPCData(IPCData&&);
    WEBCORE_EXPORT IPCData ipcData() const;

private:
    GRefPtr<GTlsCertificate> m_certificate;
};

} // namespace WebCore
