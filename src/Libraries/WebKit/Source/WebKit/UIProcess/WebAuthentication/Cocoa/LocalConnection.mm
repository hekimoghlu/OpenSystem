/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 3, 2021.
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
#import "LocalConnection.h"

#if ENABLE(WEB_AUTHN)

#import <WebCore/LocalizedStrings.h>
#import <WebCore/UserVerificationRequirement.h>
#import <WebCore/WebAuthenticationConstants.h>
#import <wtf/BlockPtr.h>
#import <wtf/RunLoop.h>
#import <wtf/TZoneMallocInlines.h>
#import <wtf/cocoa/SpanCocoa.h>

#import "LocalAuthenticationSoftLink.h"

#if USE(APPLE_INTERNAL_SDK)
#import <WebKitAdditions/LocalConnectionAdditions.h>
#else
#define LOCAL_CONNECTION_ADDITIONS
#endif

namespace WebKit {
using namespace WebCore;

namespace {
#if PLATFORM(MAC)
static inline String bundleName()
{
    return [[NSRunningApplication currentApplication] localizedName];
}
#endif
} // namespace

WTF_MAKE_TZONE_ALLOCATED_IMPL(LocalConnection);

Ref<LocalConnection> LocalConnection::create()
{
    return adoptRef(*new LocalConnection);
}

LocalConnection::~LocalConnection()
{
    // Dismiss any showing LocalAuthentication dialogs.
    [m_context invalidate];
}

void LocalConnection::verifyUser(const String& rpId, ClientDataType type, SecAccessControlRef accessControl, UserVerificationRequirement uv, UserVerificationCallback&& completionHandler)
{
    String title = genericTouchIDPromptTitle();
#if PLATFORM(MAC)
    switch (type) {
    case ClientDataType::Create:
        title = makeCredentialTouchIDPromptTitle(bundleName(), rpId);
        break;
    case ClientDataType::Get:
        title = getAssertionTouchIDPromptTitle(bundleName(), rpId);
        break;
    default:
        ASSERT_NOT_REACHED();
    }
#endif

    m_context = adoptNS([allocLAContextInstance() init]);

    auto options = adoptNS([[NSMutableDictionary alloc] init]);
#if HAVE(UNIFIED_ASC_AUTH_UI)
    if ([m_context biometryType] == LABiometryTypeTouchID) {
        [options setObject:title forKey:@(LAOptionAuthenticationTitle)];
        [options setObject:@NO forKey:@(LAOptionFallbackVisible)];
    }
#endif

    auto reply = makeBlockPtr([context = m_context, completionHandler = WTFMove(completionHandler)] (NSDictionary *information, NSError *error) mutable {
        UserVerification verification = UserVerification::Yes;
        if (error) {
            LOG_ERROR("Couldn't authenticate with biometrics: %@", error);
            verification = UserVerification::No;
            if (error.code == LAErrorUserCancel)
                verification = UserVerification::Cancel;
        }
        if (information[@"UserPresence"])
            verification = UserVerification::Presence;

        // This block can be executed in another thread.
        RunLoop::main().dispatch([completionHandler = WTFMove(completionHandler), verification, context = WTFMove(context)] () mutable {
            completionHandler(verification, context.get());
        });
    });

#if USE(APPLE_INTERNAL_SDK)
    // Depending on certain internal requirements, accessControl might not require user verifications.
    // Hence, here introduces a quirk to force the compatible mode to require user verifications if necessary.
    if (shouldUseAlternateAttributes()) {
        NSError *error = nil;
        auto canEvaluatePolicy = [m_context canEvaluatePolicy:LAPolicyDeviceOwnerAuthenticationWithBiometrics error:&error];
        if (error.code == LAErrorBiometryLockout)
            canEvaluatePolicy = true;

        if (uv == UserVerificationRequirement::Required || canEvaluatePolicy) {
            [m_context evaluatePolicy:LAPolicyDeviceOwnerAuthentication options:options.get() reply:reply.get()];
            return;
        }

        reply(@{ @"UserPresence": @YES }, nullptr);
        return;
    }
#endif

    [m_context evaluateAccessControl:accessControl operation:LAAccessControlOperationUseKeySign options:options.get() reply:reply.get()];
}

void LocalConnection::verifyUser(SecAccessControlRef accessControl, LAContext *context, CompletionHandler<void(UserVerification)>&& completionHandler)
{
    auto options = adoptNS([[NSMutableDictionary alloc] init]);
    [options setObject:@YES forKey:@(LAOptionNotInteractive)];

    auto reply = makeBlockPtr([completionHandler = WTFMove(completionHandler)] (NSDictionary *information, NSError *error) mutable {
        UserVerification verification = UserVerification::Yes;
        if (error) {
            LOG_ERROR("Couldn't authenticate with biometrics: %@", error);
            verification = UserVerification::No;
            if (error.code == LAErrorUserCancel)
                verification = UserVerification::Cancel;
        }
        if (information[@"UserPresence"])
            verification = UserVerification::Presence;

        // This block can be executed in another thread.
        RunLoop::main().dispatch([completionHandler = WTFMove(completionHandler), verification] () mutable {
            completionHandler(verification);
        });
    });

#if USE(APPLE_INTERNAL_SDK)
    // Depending on certain internal requirements, context might be nil. In that case, just check user presence.
    if (shouldUseAlternateAttributes() && !context) {
        reply(@{ @"UserPresence": @YES }, nullptr);
        return;
    }

#if PLATFORM(IOS_FAMILY_SIMULATOR)
    // Simulator doesn't support LAAccessControlOperationUseKeySign, but does support alternate attributes.
    if (shouldUseAlternateAttributes()) {
        reply(@{ }, nullptr);
        return;
    }
#endif
#endif // USE(APPLE_INTERNAL_SDK)

    [context evaluateAccessControl:accessControl operation:LAAccessControlOperationUseKeySign options:options.get() reply:reply.get()];
}

RetainPtr<SecKeyRef> LocalConnection::createCredentialPrivateKey(LAContext *context, SecAccessControlRef accessControlRef, const String& secAttrLabel, NSData *secAttrApplicationTag) const
{
    RetainPtr privateKeyAttributes = @{
        (id)kSecAttrAccessControl: (id)accessControlRef,
        (id)kSecAttrIsPermanent: @YES,
        (id)kSecAttrAccessGroup: LocalAuthenticatorAccessGroup,
        (id)kSecAttrLabel: secAttrLabel,
        (id)kSecAttrApplicationTag: secAttrApplicationTag,
    };

    if (context) {
        auto mutableCopy = adoptNS([privateKeyAttributes mutableCopy]);
        mutableCopy.get()[(id)kSecUseAuthenticationContext] = context;
        privateKeyAttributes = WTFMove(mutableCopy);
    }

    NSDictionary *attributes = @{
        (id)kSecAttrTokenID: (id)kSecAttrTokenIDSecureEnclave,
        (id)kSecAttrKeyType: (id)kSecAttrKeyTypeECSECPrimeRandom,
        (id)kSecAttrKeySizeInBits: @256,
        (id)kSecPrivateKeyAttrs: privateKeyAttributes.get(),
    };

    LOCAL_CONNECTION_ADDITIONS
    CFErrorRef errorRef = nullptr;
    auto credentialPrivateKey = adoptCF(SecKeyCreateRandomKey((__bridge CFDictionaryRef)attributes, &errorRef));
    auto retainError = adoptCF(errorRef);
    if (errorRef) {
        LOG_ERROR("Couldn't create private key: %@", (NSError *)errorRef);
        return nullptr;
    }
    return credentialPrivateKey;
}

RetainPtr<NSArray> LocalConnection::getExistingCredentials(const String& rpId)
{
    // Search Keychain for existing credential matched the RP ID.
    NSDictionary *query = @{
        (id)kSecClass: (id)kSecClassKey,
        (id)kSecAttrSynchronizable: (id)kSecAttrSynchronizableAny,
        (id)kSecAttrAccessGroup: LocalAuthenticatorAccessGroup,
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
