/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 19, 2024.
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
#ifndef _SECURITY_TRUSTDFILELOCATIONS_H_
#define _SECURITY_TRUSTDFILELOCATIONS_H_

#include <sys/types.h>
#include <CoreFoundation/CFData.h>
#include <CoreFoundation/CFString.h>
#include <CoreFoundation/CFURL.h>
#include "utilities/SecFileLocations.h"

__BEGIN_DECLS

#define TRUSTD_ROLE_ACCOUNT 282

#define TRUST_SETTINGS_STAFF_GID    20
#define TRUST_SETTINGS_USER_MODE    0600    /* owner can read/write, no others have access */
#define TRUST_SETTINGS_ADMIN_MODE   0666    /* writable only if entitled, but as any uid */

// Utility functions to return uuid for the supplied uid
CFStringRef SecCopyUUIDStringForUID(uid_t uid);
CFDataRef SecCopyUUIDDataForUID(uid_t uid);

// Returns a boolean for whether the current instance is the system trustd
bool SecOTAPKIIsSystemTrustd(void);

CFURLRef SecCopyURLForFileInRevocationInfoDirectory(CFStringRef fileName) CF_RETURNS_RETAINED;
CFURLRef SecCopyURLForFileInProtectedTrustdDirectory(CFStringRef fileName) CF_RETURNS_RETAINED;
CFURLRef SecCopyURLForFileInPrivateTrustdDirectory(CFStringRef fileName) CF_RETURNS_RETAINED;
CFURLRef SecCopyURLForFileInPrivateUserTrustdDirectory(CFStringRef fileName) CF_RETURNS_RETAINED;

void WithPathInRevocationInfoDirectory(CFStringRef fileName, void(^operation)(const char *utf8String));
void WithPathInProtectedTrustdDirectory(CFStringRef fileName, void(^operation)(const char *utf8String));
void WithPathInPrivateTrustdDirectory(CFStringRef fileName, void(^operation)(const char *utf8String));
void WithPathInPrivateUserTrustdDirectory(CFStringRef fileName, void(^operation)(const char *utf8String));

void FixTrustdFilePermissions(void);
bool TrustdChangeFileProtectionToClassD(const char *filename, CFErrorRef *error);

#if __OBJC__
#define TrustdFileHelperXPCServiceName "com.apple.trustdFileHelper"
@protocol TrustdFileHelper_protocol
- (void)fixFiles:(void (^)(BOOL, NSError*))reply;
@end

@interface NSDictionary (trustdAdditions)
- (BOOL)writeToClassDURL:(NSURL *)url permissions:(mode_t)permissions error:(NSError **)error;
@end
#endif  // __OBJC__

__END_DECLS

#endif /* _SECURITY_TRUSTDFILELOCATIONS_H_ */
