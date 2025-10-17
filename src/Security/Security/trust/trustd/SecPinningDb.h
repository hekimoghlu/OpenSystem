/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 28, 2022.
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
 @header SecPinningDb
 The functions in SecPinningDb.h provide an interface to look up
 pinning rules based on hostname.
 */

#ifndef _SECURITY_SECPINNINGDB_H_
#define _SECURITY_SECPINNINGDB_H_

#include <CoreFoundation/CoreFoundation.h>
#include <utilities/SecDb.h>

__BEGIN_DECLS

CF_ASSUME_NONNULL_BEGIN
CF_IMPLICIT_BRIDGING_ENABLED

extern const uint64_t PinningDbSchemaVersion;

extern const CFStringRef kSecPinningDbKeyHostname;
extern const CFStringRef kSecPinningDbKeyPolicyName;
extern const CFStringRef kSecPinningDbKeyRules;
extern const CFStringRef kSecPinningDbKeyTransparentConnection;

CFDictionaryRef _Nullable SecPinningDbCopyMatching(CFDictionaryRef _Nonnull query);
void SecPinningDbInitialize(void);

#if __OBJC__
bool SecPinningDbUpdateFromURL(NSURL *url, NSError **error);

@interface SecPinningDb : NSObject
@property (assign) SecDbRef db;
+ (NSURL *)pinningDbPath;
- (NSNumber *)getContentVersion:(SecDbConnectionRef)dbconn error:(CFErrorRef *)error;
- (NSNumber *)getSchemaVersion:(SecDbConnectionRef)dbconn error:(CFErrorRef *)error;
- (BOOL) installDbFromURL:(NSURL *)localURL error:(NSError **)nserror;
@end
#endif // __OBJC__

CFNumberRef SecPinningDbCopyContentVersion(void);

CF_IMPLICIT_BRIDGING_DISABLED
CF_ASSUME_NONNULL_END

__END_DECLS


#endif /* _SECURITY_SECPINNINGDB_H_ */
