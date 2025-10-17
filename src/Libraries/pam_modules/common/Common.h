/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 3, 2025.
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

#include <OpenDirectory/OpenDirectory.h>
#include <security/pam_appl.h>
#include <security/pam_modules.h>

#ifndef _COMMON_H_
#define _COMMON_H_

int od_record_create(pam_handle_t*, ODRecordRef*, CFStringRef);
int od_record_create_cstring(pam_handle_t*, ODRecordRef*, const char*);
int od_record_attribute_create_cfstring(ODRecordRef record, CFStringRef attrib,  CFStringRef *out);
int od_record_attribute_create_cfarray(ODRecordRef record, CFStringRef attrib,  CFArrayRef *out);
int od_record_attribute_create_cstring(ODRecordRef record, CFStringRef attrib,  char **out);

int od_record_check_pwpolicy(ODRecordRef);
int od_record_check_authauthority(ODRecordRef);
int od_record_check_homedir(ODRecordRef);
int od_record_check_shell(ODRecordRef);

int od_extract_home(pam_handle_t*, const char *, char **, char **, char **);
int od_principal_for_user(pam_handle_t*, const char *, char **);

void pam_cf_cleanup(__unused pam_handle_t *, void *, __unused int );

int cfstring_to_cstring(const CFStringRef val, char **buffer);

#ifndef CFReleaseSafe
#define CFReleaseSafe(CF) { CFTypeRef _cf = (CF); if (_cf) CFRelease(_cf); }
#endif

#ifndef CFReleaseNull
#define CFReleaseNull(CF) { CFTypeRef _cf = (CF); \
if (_cf) { (CF) = NULL; CFRelease(_cf); } }
#endif

CF_RETURNS_RETAINED
CFStringRef    GetPrincipalFromUser(CFDictionaryRef inUserRecord);

#endif /* _COMMON_H_ */
