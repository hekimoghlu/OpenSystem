/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 24, 2022.
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
#import "MockLocalConnection.h"

#if ENABLE(WEB_AUTHN)

#import <JavaScriptCore/ArrayBuffer.h>
#import <Security/SecItem.h>
#import <WebCore/AuthenticatorAssertionResponse.h>
#import <WebCore/ExceptionData.h>
#import <wtf/RunLoop.h>
#import <wtf/TZoneMallocInlines.h>
#import <wtf/spi/cocoa/SecuritySPI.h>
#import <wtf/text/Base64.h>
#import <wtf/text/WTFString.h>

#import "LocalAuthenticationSoftLink.h"

namespace WebKit {
using namespace WebCore;

WTF_MAKE_TZONE_ALLOCATED_IMPL(MockLocalConnection);

Ref<MockLocalConnection> MockLocalConnection::create(const WebCore::MockWebAuthenticationConfiguration& configuration)
{
    return adoptRef(*new MockLocalConnection(configuration));
}

MockLocalConnection::MockLocalConnection(const MockWebAuthenticationConfiguration& configuration)
    : m_configuration(configuration)
{
}

void MockLocalConnection::verifyUser(const String&, ClientDataType, SecAccessControlRef, WebCore::UserVerificationRequirement, UserVerificationCallback&& callback)
{
    // Mock async operations.
    RunLoop::main().dispatch([configuration = m_configuration, callback = WTFMove(callback)]() mutable {
        ASSERT(configuration.local);

        UserVerification userVerification = UserVerification::No;
        switch (configuration.local->userVerification) {
        case MockWebAuthenticationConfiguration::UserVerification::No:
            break;
        case MockWebAuthenticationConfiguration::UserVerification::Yes:
            userVerification = UserVerification::Yes;
            break;
        case MockWebAuthenticationConfiguration::UserVerification::Cancel:
            userVerification = UserVerification::Cancel;
            break;
        case MockWebAuthenticationConfiguration::UserVerification::Presence:
            userVerification = UserVerification::Presence;
            break;
        }

        callback(userVerification, adoptNS([allocLAContextInstance() init]).get());
    });
}

void MockLocalConnection::verifyUser(SecAccessControlRef, LAContext *, CompletionHandler<void(UserVerification)>&& callback)
{
    // Mock async operations.
    RunLoop::main().dispatch([configuration = m_configuration, callback = WTFMove(callback)]() mutable {
        ASSERT(configuration.local);

        UserVerification userVerification = UserVerification::No;
        switch (configuration.local->userVerification) {
        case MockWebAuthenticationConfiguration::UserVerification::No:
            break;
        case MockWebAuthenticationConfiguration::UserVerification::Yes:
            userVerification = UserVerification::Yes;
            break;
        case MockWebAuthenticationConfiguration::UserVerification::Cancel:
            userVerification = UserVerification::Cancel;
            break;
        case MockWebAuthenticationConfiguration::UserVerification::Presence:
            userVerification = UserVerification::Presence;
            break;
        }

        callback(userVerification);
    });
}

RetainPtr<SecKeyRef> MockLocalConnection::createCredentialPrivateKey(LAContext *, SecAccessControlRef, const String& secAttrLabel, NSData *secAttrApplicationTag) const
{
    ASSERT(m_configuration.local);

    // Get Key and add it to Keychain.
    NSDictionary* options = @{
        (id)kSecAttrKeyType: (id)kSecAttrKeyTypeECSECPrimeRandom,
        (id)kSecAttrKeyClass: (id)kSecAttrKeyClassPrivate,
        (id)kSecAttrKeySizeInBits: @256,
    };
    CFErrorRef errorRef = nullptr;
    auto key = adoptCF(SecKeyCreateWithData(
        (__bridge CFDataRef)adoptNS([[NSData alloc] initWithBase64EncodedString:m_configuration.local->privateKeyBase64 options:NSDataBase64DecodingIgnoreUnknownCharacters]).get(),
        (__bridge CFDictionaryRef)options,
        &errorRef
    ));
    if (errorRef)
        return nullptr;

    NSDictionary* addQuery = @{
        (id)kSecValueRef: (id)key.get(),
        (id)kSecClass: (id)kSecClassKey,
        (id)kSecAttrLabel: secAttrLabel,
        (id)kSecAttrApplicationTag: secAttrApplicationTag,
        (id)kSecAttrAccessible: (id)kSecAttrAccessibleAfterFirstUnlock,
        (id)kSecUseDataProtectionKeychain: @YES
    };
    OSStatus status = SecItemAdd((__bridge CFDictionaryRef)addQuery, NULL);
    if (status) {
        LOG_ERROR("Couldn't add the key to the keychain. %d", status);
        return nullptr;
    }

    return key;
}

void MockLocalConnection::filterResponses(Vector<Ref<AuthenticatorAssertionResponse>>& responses) const
{
    const auto& preferredCredentialIdBase64 = m_configuration.local->preferredCredentialIdBase64;
    if (preferredCredentialIdBase64.isEmpty())
        return;

    RefPtr<AuthenticatorAssertionResponse> matchingResponse;
    for (auto& response : responses) {
        auto* rawId = response->rawId();
        ASSERT(rawId);
        auto rawIdBase64 = base64EncodeToString(rawId->span());
        if (rawIdBase64 == preferredCredentialIdBase64) {
            matchingResponse = response.copyRef();
            break;
        }
    }
    responses.clear();
    responses.append(matchingResponse.releaseNonNull());
}

RetainPtr<NSArray> MockLocalConnection::getExistingCredentials(const String& rpId)
{
    // Search Keychain for existing credential matched the RP ID.
    NSDictionary *query = @{
        (id)kSecClass: (id)kSecClassKey,
        (id)kSecAttrKeyClass: (id)kSecAttrKeyClassPrivate,
        (id)kSecAttrSynchronizable: (id)kSecAttrSynchronizableAny,
        (id)kSecAttrLabel: rpId,
        (id)kSecReturnAttributes: @YES,
        (id)kSecMatchLimit: (id)kSecMatchLimitAll,
        (id)kSecUseDataProtectionKeychain: @YES
    };

    CFTypeRef attributesArrayRef = nullptr;
    OSStatus status = SecItemCopyMatching((__bridge CFDictionaryRef)query, &attributesArrayRef);
    if (status && status != errSecItemNotFound)
        return nullptr;
    auto retainAttributesArray = adoptCF(attributesArrayRef);
    NSArray *sortedAttributesArray = [(NSArray *)attributesArrayRef sortedArrayUsingComparator:^(NSDictionary *a, NSDictionary *b) {
        return [b[(id)kSecAttrModificationDate] compare:a[(id)kSecAttrModificationDate]];
    }];
    return retainPtr(sortedAttributesArray);
}

} // namespace WebKit

#endif // ENABLE(WEB_AUTHN)
