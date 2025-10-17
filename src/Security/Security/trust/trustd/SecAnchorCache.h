/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 25, 2024.
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
/*!
 @header SecAnchorCache
 The functions provided in SecAnchorCache.h provide an interface to
 a caching module for SecAnchorRef instances. Lookups which do not hit
 the cache will attempt to read the certificate data from the disk and
 then add a new entry to the cache.
 */

#ifndef _SECURITY_SECANCHORCACHE_H_
#define _SECURITY_SECANCHORCACHE_H_

#include <CoreFoundation/CoreFoundation.h>
#include <Security/Security.h>

#include <os/transaction_private.h>
#include <os/variant_private.h>
#include <os/lock.h>

#if __OBJC__
#import <Foundation/Foundation.h>
#endif

__BEGIN_DECLS

CF_ASSUME_NONNULL_BEGIN
CF_IMPLICIT_BRIDGING_ENABLED

extern CFStringRef kSecAnchorTypeUnspecified;
extern CFStringRef kSecAnchorTypeSystem;
extern CFStringRef kSecAnchorTypePlatform;
extern CFStringRef kSecAnchorTypeCustom;

#if __OBJC__

@interface SecAnchorCache : NSObject

- (SecCertificateRef _Nullable)copyAnchorAssetForKey:(NSString * _Nullable)anchorHash;
- (NSArray * _Nonnull)anchorsForKey:(NSString*_Nullable)anchorLookupKey;

@end
#endif // __OBJC__

void SecAnchorCacheInitialize(void);
CFArrayRef SecAnchorCacheCopyParentCertificates(CFStringRef anchorLookupKey);
CFArrayRef SecAnchorCacheCopyAnchors(CFStringRef policyId);

bool SecAnchorPolicyPermitsAnchorRecord(CFDictionaryRef cfAnchorRecord, CFStringRef policyId);

CF_IMPLICIT_BRIDGING_DISABLED
CF_ASSUME_NONNULL_END

__END_DECLS

#endif /* _SECURITY_SECANCHORCACHE_H_ */
