/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 6, 2022.
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
#ifndef _SECURITY_AUTH_UTILITIES_H_
#define _SECURITY_AUTH_UTILITIES_H_

#include <xpc/xpc.h>
#include <CoreFoundation/CoreFoundation.h>
#include <Security/Authorization.h>

#if defined(__cplusplus)
extern "C" {
#endif

CF_RETURNS_RETAINED AuthorizationItemSet * DeserializeItemSet(const xpc_object_t);
XPC_RETURNS_RETAINED xpc_object_t SerializeItemSet(const AuthorizationItemSet*);
void FreeItemSet(AuthorizationItemSet*);

char * _copy_cf_string(CFTypeRef,const char*);
int64_t _get_cf_int(CFTypeRef,int64_t);
bool _get_cf_bool(CFTypeRef,bool);

bool _compare_string(const char *, const char *);
char * _copy_string(const char *);
void * _copy_data(const void * data, size_t dataLen);

bool _cf_set_iterate(CFSetRef, bool(^iterator)(CFTypeRef value));
bool _cf_bag_iterate(CFBagRef, bool(^iterator)(CFTypeRef value));
bool _cf_dictionary_iterate(CFDictionaryRef, bool(^iterator)(CFTypeRef key,CFTypeRef value));

bool isInFVUnlockOrRecovery(void);

#if defined(__cplusplus)
}
#endif

#endif /* !_SECURITY_AUTH_UTILITIES_H_ */
