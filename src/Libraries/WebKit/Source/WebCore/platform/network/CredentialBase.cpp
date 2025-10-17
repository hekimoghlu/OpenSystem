/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 30, 2023.
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
#include "CredentialBase.h"

#include "Credential.h"
#include <wtf/text/Base64.h>
#include <wtf/text/MakeString.h>

namespace WebCore {

// Need to enforce empty, non-null strings due to the pickiness of the String == String operator
// combined with the semantics of the String(NSString*) constructor
CredentialBase::CredentialBase()
    : m_user(emptyString())
    , m_password(emptyString())
    , m_persistence(CredentialPersistence::None)
{
}
   
// Need to enforce empty, non-null strings due to the pickiness of the String == String operator
// combined with the semantics of the String(NSString*) constructor
CredentialBase::CredentialBase(const String& user, const String& password, CredentialPersistence persistence)
    : m_user(user.length() ? user : emptyString())
    , m_password(password.length() ? password : emptyString())
    , m_persistence(persistence)
{
}
    
CredentialBase::CredentialBase(const Credential& original, CredentialPersistence persistence)
    : m_user(original.user())
    , m_password(original.password())
    , m_persistence(persistence)
{
}

bool CredentialBase::isEmpty() const
{
    return m_user.isEmpty() && m_password.isEmpty();
}
    
const String& CredentialBase::user() const
{ 
    return m_user; 
}

const String& CredentialBase::password() const
{ 
    return m_password; 
}

bool CredentialBase::hasPassword() const
{ 
    return !m_password.isEmpty(); 
}

CredentialPersistence CredentialBase::persistence() const
{ 
    return m_persistence; 
}

bool CredentialBase::compare(const Credential& a, const Credential& b)
{
    // Check persistence first since all credential types
    // have the persistence property.
    if (a.persistence() != b.persistence())
        return false;
    if (a.user() != b.user())
        return false;
    if (a.password() != b.password())
        return false;
        
    return Credential::platformCompare(a, b);
}

String CredentialBase::serializationForBasicAuthorizationHeader() const
{
    auto credentialStringData = makeString(m_user, ':', m_password).utf8();
    return makeString("Basic "_s, base64Encoded(credentialStringData.span()));
}

auto CredentialBase::nonPlatformData() const -> NonPlatformData
{
    return {
        user(),
        password(),
        persistence()
    };
}

} // namespace WebCore
