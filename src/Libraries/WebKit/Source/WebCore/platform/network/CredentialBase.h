/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 11, 2023.
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

#include <wtf/text/WTFString.h>

namespace WebCore {

class Credential;

enum class CredentialPersistence : uint8_t {
    None,
    ForSession,
    Permanent
};

class CredentialBase {
public:
    WEBCORE_EXPORT bool isEmpty() const;
    
    WEBCORE_EXPORT const String& user() const;
    WEBCORE_EXPORT const String& password() const;
    WEBCORE_EXPORT bool hasPassword() const;
    WEBCORE_EXPORT CredentialPersistence persistence() const;

    bool encodingRequiresPlatformData() const { return false; }

    WEBCORE_EXPORT static bool compare(const Credential&, const Credential&);

    WEBCORE_EXPORT String serializationForBasicAuthorizationHeader() const;

    struct NonPlatformData {
        String user;
        String password;
        CredentialPersistence persistence;
    };

    WEBCORE_EXPORT NonPlatformData nonPlatformData() const;

protected:
    WEBCORE_EXPORT CredentialBase();
    WEBCORE_EXPORT CredentialBase(const String& user, const String& password, CredentialPersistence);
    CredentialBase(const Credential& original, CredentialPersistence);

    static bool platformCompare(const Credential&, const Credential&) { return true; }

private:
    String m_user;
    String m_password;
    CredentialPersistence m_persistence;
};

inline bool operator==(const Credential& a, const Credential& b) { return CredentialBase::compare(a, b); }
    
} // namespace WebCore
