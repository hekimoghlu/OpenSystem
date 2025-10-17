/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 7, 2023.
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
//  PreloginUserDb.h
//  authd
//

OSStatus preloginudb_copy_userdb(const char * _Nullable uuid, UInt32 flags, CFArrayRef _Nonnull * _Nonnull output);
OSStatus prelogin_copy_pref_value(const char * _Nullable uuid, const char * _Nullable user, const char * _Nonnull domain, const char * _Nonnull item, CFTypeRef _Nonnull * _Nonnull output);
OSStatus prelogin_smartcardonly_override(const char * _Nullable uuid, unsigned char operation, Boolean * _Nullable status);
Boolean prelogin_reset_db_wanted(void);

Boolean pssoEnabled(void);

CF_RETURNS_RETAINED CFDateRef _Nonnull dateFromUnixTimestamp(__darwin_time_t seconds);
CF_RETURNS_RETAINED CFDictionaryRef _Nullable readDatabasePlist(CFStringRef _Nonnull path, CFErrorRef _Nullable * _Nullable error);
