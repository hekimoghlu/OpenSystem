/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 1, 2023.
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

#include "CoreIPCDate.h"
#include "CoreIPCNumber.h"
#include "CoreIPCSecTrust.h"
#include "CoreIPCString.h"

#include <wtf/ArgumentCoder.h>
#include <wtf/RetainPtr.h>
#include <wtf/TZoneMalloc.h>

OBJC_CLASS NSURLCredential;

namespace WebKit {

#if HAVE(WK_SECURE_CODING_NSURLCREDENTIAL)

enum class CoreIPCNSURLCredentialPersistence : uint8_t {
    None = 1,
    Session,
    Permanent,
    Synchronizable
};

enum class CoreIPCNSURLCredentialType : uint8_t {
    Password,
    ServerTrust,
    KerberosTicket,
    ClientCertificate,
    XMobileMeAuthToken,
    OAuth2
};

struct CoreIPCNSURLCredentialData {
    using Flags = std::pair<CoreIPCString, CoreIPCString>;
    using Attributes = std::pair<CoreIPCString, std::variant<CoreIPCNumber, CoreIPCString, CoreIPCDate>>;

    CoreIPCNSURLCredentialPersistence persistence { CoreIPCNSURLCredentialPersistence::None };
    CoreIPCNSURLCredentialType type { CoreIPCNSURLCredentialType::Password };
    std::optional<CoreIPCString> user;
    std::optional<CoreIPCString> password;
    std::optional<Vector<Attributes>> attributes;
    std::optional<CoreIPCString> identifier;
    std::optional<bool> useKeychain;
    CoreIPCSecTrust trust;
    std::optional<CoreIPCString> service;
    std::optional<Vector<Flags>> flags;
    std::optional<CoreIPCString> uuid;
    std::optional<CoreIPCString> appleID;
    std::optional<CoreIPCString> realm;
    std::optional<CoreIPCString> token;
};

class CoreIPCNSURLCredential {
    WTF_MAKE_TZONE_ALLOCATED(CoreIPCNSURLCredential);
public:
    CoreIPCNSURLCredential(NSURLCredential *);
    CoreIPCNSURLCredential(CoreIPCNSURLCredentialData&&);

    RetainPtr<id> toID() const;
private:
    friend struct IPC::ArgumentCoder<CoreIPCNSURLCredential, void>;
    CoreIPCNSURLCredentialData m_data;
};

#endif

#if !HAVE(WK_SECURE_CODING_NSURLCREDENTIAL) && !HAVE(DICTIONARY_SERIALIZABLE_NSURLCREDENTIAL)

class CoreIPCNSURLCredential {
public:
    CoreIPCNSURLCredential(NSURLCredential *);
    CoreIPCNSURLCredential(const RetainPtr<NSURLCredential>& credential)
        : CoreIPCNSURLCredential(credential.get()) { }
    CoreIPCNSURLCredential(RetainPtr<NSData>&& serializedBytes)
        : m_serializedBytes(WTFMove(serializedBytes)) { }

    RetainPtr<id> toID() const;

private:
    friend struct IPC::ArgumentCoder<CoreIPCNSURLCredential, void>;

    RetainPtr<NSData> m_serializedBytes;
};

#endif

} // namespace WebKit
