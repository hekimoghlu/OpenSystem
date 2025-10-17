/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 11, 2022.
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
//
//  SecFileLocations.h
//  utilities
//


#ifndef _SECFILELOCATIONS_H_
#define _SECFILELOCATIONS_H_

#include <CoreFoundation/CFURL.h>
#include <TargetConditionals.h>

__BEGIN_DECLS

#if TARGET_OS_IOS
bool SecSupportsEnhancedApfs(void);
#endif
bool SecIsEduMode(void);
bool SecSeparateUserKeychain(void);

CFURLRef SecCopyURLForFileInBaseDirectory(bool system, CFStringRef directoryPath, CFStringRef fileName) CF_RETURNS_RETAINED;
CFURLRef SecCopyURLForFileInKeychainDirectory(CFStringRef fileName) CF_RETURNS_RETAINED;
CFURLRef SecCopyURLForFileInUserScopedKeychainDirectory(CFStringRef fileName) CF_RETURNS_RETAINED;
CFURLRef SecCopyURLForFileInSystemKeychainDirectory(CFStringRef fileName) CF_RETURNS_RETAINED;
CFURLRef SecCopyURLForFileInUserCacheDirectory(CFStringRef fileName) CF_RETURNS_RETAINED;
CFURLRef SecCopyURLForFileInPreferencesDirectory(CFStringRef fileName) CF_RETURNS_RETAINED;
CFURLRef SecCopyURLForFileInManagedPreferencesDirectory(CFStringRef fileName) CF_RETURNS_RETAINED;
CFURLRef SecCopyURLForFileInProtectedDirectory(CFStringRef fileName) CF_RETURNS_RETAINED;

void WithPathInDirectory(CFURLRef fileURL, void(^operation)(const char *utf8String));
void WithPathInKeychainDirectory(CFStringRef fileName, void(^operation)(const char *utf8String));
void WithPathInUserCacheDirectory(CFStringRef fileName, void(^operation)(const char *utf8String));
void WithPathInProtectedDirectory(CFStringRef fileName, void(^operation)(const char *utf8String));

void SetCustomHomePath(const char* path);
void SecSetCustomHomeURLString(CFStringRef path);
void SecSetCustomHomeURL(CFURLRef url);

CFURLRef SecCopyHomeURL(void) CF_RETURNS_RETAINED;

__END_DECLS

#endif
