/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 5, 2023.
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
#import "CredentialCocoa.h"

namespace WebCore {

static NSURLCredentialPersistence toNSURLCredentialPersistence(CredentialPersistence persistence)
{
    switch (persistence) {
    case CredentialPersistence::None:
        return NSURLCredentialPersistenceNone;
    case CredentialPersistence::ForSession:
        return NSURLCredentialPersistenceForSession;
    case CredentialPersistence::Permanent:
        return NSURLCredentialPersistencePermanent;
    }

    ASSERT_NOT_REACHED();
    return NSURLCredentialPersistenceNone;
}

static CredentialPersistence toCredentialPersistence(NSURLCredentialPersistence persistence)
{
    switch (persistence) {
    case NSURLCredentialPersistenceNone:
        return CredentialPersistence::None;
    case NSURLCredentialPersistenceForSession:
        return CredentialPersistence::ForSession;
    case NSURLCredentialPersistencePermanent:
    case NSURLCredentialPersistenceSynchronizable:
        return CredentialPersistence::Permanent;
    }

    ASSERT_NOT_REACHED();
    return CredentialPersistence::None;
}

Credential::Credential(const Credential& original, CredentialPersistence persistence)
    : CredentialBase(original, persistence)
{
    NSURLCredential *originalNSURLCredential = original.m_nsCredential.get();
    if (!originalNSURLCredential)
        return;

    if (NSString *user = originalNSURLCredential.user)
        m_nsCredential = adoptNS([[NSURLCredential alloc] initWithUser:user password:originalNSURLCredential.password persistence:toNSURLCredentialPersistence(persistence)]);
    else if (SecIdentityRef identity = originalNSURLCredential.identity)
        m_nsCredential = adoptNS([[NSURLCredential alloc] initWithIdentity:identity certificates:originalNSURLCredential.certificates persistence:toNSURLCredentialPersistence(persistence)]);
    else {
        // It is not possible to set the persistence of server trust credentials.
        ASSERT_NOT_REACHED();
        m_nsCredential = originalNSURLCredential;
    }
}

Credential::Credential(NSURLCredential *credential)
    : CredentialBase(credential.user, credential.password, toCredentialPersistence(credential.persistence))
    , m_nsCredential(credential)
{
}

NSURLCredential *Credential::nsCredential() const
{
    if (m_nsCredential)
        return m_nsCredential.get();

    if (CredentialBase::isEmpty())
        return nil;

    m_nsCredential = adoptNS([[NSURLCredential alloc] initWithUser:user() password:password() persistence:toNSURLCredentialPersistence(persistence())]);

    return m_nsCredential.get();
}

bool Credential::isEmpty() const
{
    if (m_nsCredential)
        return false;

    return CredentialBase::isEmpty();
}

bool Credential::platformCompare(const Credential& a, const Credential& b)
{
    if (!a.m_nsCredential && !b.m_nsCredential)
        return true;

    return [a.nsCredential() isEqual:b.nsCredential()];
}

bool Credential::encodingRequiresPlatformData(NSURLCredential *credential)
{
    return !credential.user;
}

Credential Credential::fromIPCData(IPCData&& ipcData)
{
    return WTF::switchOn(WTFMove(ipcData), [](NonPlatformData&& data) {
        return Credential { data.user, data.password, data.persistence };
    }, [](RetainPtr<NSURLCredential>&& credential) {
        return Credential { credential.get() };
    });
}

auto Credential::ipcData() const -> IPCData
{
    if (encodingRequiresPlatformData())
        return m_nsCredential;
    return nonPlatformData();
}

} // namespace WebCore
