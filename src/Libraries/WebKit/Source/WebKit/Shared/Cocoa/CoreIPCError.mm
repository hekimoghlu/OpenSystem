/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 17, 2022.
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
#import "CoreIPCError.h"

#import <wtf/TZoneMallocInlines.h>
#import <wtf/URL.h>
#import <wtf/cocoa/TypeCastsCocoa.h>

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(CoreIPCError);

bool CoreIPCError::hasValidUserInfo(const RetainPtr<CFDictionaryRef>& userInfo)
{
    NSDictionary * info = bridge_cast(userInfo.get());

    if (id object = [info objectForKey:@"NSErrorClientCertificateChainKey"]) {
        if (![object isKindOfClass:[NSArray class]])
            return false;
        for (id certificate in object) {
            if ((CFGetTypeID((__bridge CFTypeRef)certificate) != SecCertificateGetTypeID()))
                return false;
        }
    }

    if (id peerCertificateChain = [info objectForKey:@"NSErrorPeerCertificateChainKey"]) {
        for (id object in peerCertificateChain) {
            if (CFGetTypeID((__bridge CFTypeRef)object) != SecCertificateGetTypeID())
                return false;
        }
    }

    if (SecTrustRef peerTrust = (__bridge SecTrustRef)[info objectForKey:NSURLErrorFailingURLPeerTrustErrorKey]) {
        if (CFGetTypeID((__bridge CFTypeRef)peerTrust) != SecTrustGetTypeID())
            return false;
    }

    if (id underlyingError = [info objectForKey:NSUnderlyingErrorKey]) {
        if (![underlyingError isKindOfClass:[NSError class]])
            return false;
    }

    return true;
}

RetainPtr<id> CoreIPCError::toID() const
{
    if (m_underlyingError) {
        auto underlyingNSError = m_underlyingError->toID();
        if (!underlyingNSError)
            return nil;

        auto mutableUserInfo = adoptCF(CFDictionaryCreateMutableCopy(kCFAllocatorDefault, CFDictionaryGetCount(m_userInfo.get()) + 1, m_userInfo.get()));
        CFDictionarySetValue(mutableUserInfo.get(), (__bridge CFStringRef)NSUnderlyingErrorKey, (__bridge CFTypeRef)underlyingNSError.get());
        return adoptNS([[NSError alloc] initWithDomain:m_domain code:m_code userInfo:(__bridge NSDictionary *)mutableUserInfo.get()]);
    }
    return adoptNS([[NSError alloc] initWithDomain:m_domain code:m_code userInfo:(__bridge NSDictionary *)m_userInfo.get()]);
}

bool CoreIPCError::isSafeToEncodeUserInfo(id value)
{
    if ([value isKindOfClass:NSString.class] || [value isKindOfClass:NSURL.class] || [value isKindOfClass:NSNumber.class])
        return true;

    if (auto array = dynamic_objc_cast<NSArray>(value)) {
        for (id object in array) {
            if (!isSafeToEncodeUserInfo(object))
                return false;
        }
        return true;
    }

    if (auto dictionary = dynamic_objc_cast<NSDictionary>(value)) {
        for (id innerValue in dictionary.objectEnumerator) {
            if (!isSafeToEncodeUserInfo(innerValue))
                return false;
        }
        return true;
    }

    return false;
}

CoreIPCError::CoreIPCError(NSError *nsError)
    : m_domain([nsError domain])
    , m_code([nsError code])
{
    NSDictionary *userInfo = [nsError userInfo];

    RetainPtr<CFMutableDictionaryRef> filteredUserInfo = adoptCF(CFDictionaryCreateMutable(kCFAllocatorDefault, userInfo.count, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks));

    [userInfo enumerateKeysAndObjectsUsingBlock:^(id key, id value, BOOL*) {
        if ([key isEqualToString:@"NSErrorClientCertificateChainKey"]) {
            if (![value isKindOfClass:[NSArray class]])
                return;
        }
        if (isSafeToEncodeUserInfo(value))
            CFDictionarySetValue(filteredUserInfo.get(), (__bridge CFTypeRef)key, (__bridge CFTypeRef)value);
    }];

    if (NSArray *clientIdentityAndCertificates = [userInfo objectForKey:@"NSErrorClientCertificateChainKey"]) {
        if ([clientIdentityAndCertificates isKindOfClass:[NSArray class]]) {
            // Turn SecIdentity members into SecCertificate to strip out private key information.
            id clientCertificates = [NSMutableArray arrayWithCapacity:clientIdentityAndCertificates.count];
            for (id object in clientIdentityAndCertificates) {
                // Only SecIdentity or SecCertificate types are expected in clientIdentityAndCertificates
                if (CFGetTypeID((__bridge CFTypeRef)object) != SecIdentityGetTypeID() && CFGetTypeID((__bridge CFTypeRef)object) != SecCertificateGetTypeID())
                    continue;
                if (CFGetTypeID((__bridge CFTypeRef)object) != SecIdentityGetTypeID()) {
                    [clientCertificates addObject:object];
                    continue;
                }
                SecCertificateRef certificate = nil;
                OSStatus status = SecIdentityCopyCertificate((SecIdentityRef)object, &certificate);
                RetainPtr<SecCertificateRef> retainCertificate = adoptCF(certificate);
                // The SecIdentity member is the key information of this attribute. Without it, we should nil
                // the attribute.
                if (status != errSecSuccess) {
                    LOG_ERROR("Failed to encode nsError.userInfo[NSErrorClientCertificateChainKey]: %d", status);
                    clientCertificates = nil;
                    break;
                }
                [clientCertificates addObject:(__bridge id)certificate];
            }
            CFDictionarySetValue(filteredUserInfo.get(), CFSTR("NSErrorClientCertificateChainKey"), (__bridge CFTypeRef)clientCertificates);
        }
    }

    id peerCertificateChain = [userInfo objectForKey:@"NSErrorPeerCertificateChainKey"];
    if (!peerCertificateChain) {
        if (id candidatePeerTrust = [userInfo objectForKey:NSURLErrorFailingURLPeerTrustErrorKey]) {
            if (CFGetTypeID((__bridge CFTypeRef)candidatePeerTrust) == SecTrustGetTypeID())
                peerCertificateChain = (__bridge NSArray *)adoptCF(SecTrustCopyCertificateChain((__bridge SecTrustRef)candidatePeerTrust)).autorelease();
        }
    }

    if (peerCertificateChain && [peerCertificateChain isKindOfClass:[NSArray class]]) {
        bool hasExpectedTypes = true;
        for (id object in peerCertificateChain) {
            if (CFGetTypeID((__bridge CFTypeRef)object) != SecCertificateGetTypeID()) {
                hasExpectedTypes = false;
                break;
            }
        }
        if (hasExpectedTypes)
            CFDictionarySetValue(filteredUserInfo.get(), CFSTR("NSErrorPeerCertificateChainKey"), (__bridge CFTypeRef)peerCertificateChain);
    }

    if (SecTrustRef peerTrust = (__bridge SecTrustRef)[userInfo objectForKey:NSURLErrorFailingURLPeerTrustErrorKey]) {
        if (CFGetTypeID((__bridge CFTypeRef)peerTrust) == SecTrustGetTypeID())
            CFDictionarySetValue(filteredUserInfo.get(), (__bridge CFStringRef)NSURLErrorFailingURLPeerTrustErrorKey, peerTrust);
    }

    m_userInfo = static_cast<CFDictionaryRef>(filteredUserInfo.get());

    if (id underlyingError = [userInfo objectForKey:NSUnderlyingErrorKey]) {
        if ([underlyingError isKindOfClass:[NSError class]])
            m_underlyingError = makeUnique<CoreIPCError>(underlyingError);
    }
}

} // namespace WebKit
