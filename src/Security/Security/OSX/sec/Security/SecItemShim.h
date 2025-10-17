/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 17, 2022.
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
 @header SecItemShim.h
 SecItemShim defines functions and macros for shimming iOS Security
 implementation to be used inside OSX.
 */

#ifndef _SECURITY_SECITEMSHIM_H_
#define _SECURITY_SECITEMSHIM_H_

#import <Availability.h>

#if TARGET_OS_OSX

#include <CoreFoundation/CFDictionary.h>
#include <CoreFoundation/CFArray.h>
#include <CoreFoundation/CFData.h>
#include <Security/SecKey.h>

__BEGIN_DECLS

struct __SecKeyDescriptor;

OSStatus SecItemAdd_ios(CFDictionaryRef attributes, CFTypeRef *result);
OSStatus SecItemCopyMatching_ios(CFDictionaryRef query, CFTypeRef *result);
OSStatus SecItemUpdate_ios(CFDictionaryRef query, CFDictionaryRef attributesToUpdate);
OSStatus SecItemDelete_ios(CFDictionaryRef query);

OSStatus SecKeyGeneratePair_ios(CFDictionaryRef parameters, SecKeyRef *publicKey, SecKeyRef *privateKey);
SecKeyRef SecKeyCreateRandomKey_ios(CFDictionaryRef parameters, CFErrorRef *error);

#if SECITEM_SHIM_OSX

#define SecItemAdd SecItemAdd_ios
#define SecItemCopyMatching SecItemCopyMatching_ios
#define SecItemUpdate SecItemUpdate_ios
#define SecItemDelete SecItemDelete_ios

#define SecKeyGeneratePair SecKeyGeneratePair_ios
#define SecKeyCreateRandomKey SecKeyCreateRandomKey_ios

#endif

__END_DECLS

#endif // TARGET_OS_OSX
#endif /* !_SECURITY_SECITEMSHIM_H_ */
