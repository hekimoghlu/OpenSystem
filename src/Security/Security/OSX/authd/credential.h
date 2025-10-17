/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 19, 2023.
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
#ifndef _SECURITY_AUTH_CREDENTIAL_H_
#define _SECURITY_AUTH_CREDENTIAL_H_

#if defined(__cplusplus)
extern "C" {
#endif

#ifdef __BLOCKS__
    typedef bool (^credential_iterator_t)(credential_t);
#endif /* __BLOCKS__ */
    
AUTH_WARN_RESULT AUTH_MALLOC AUTH_NONNULL_ALL AUTH_RETURNS_RETAINED
credential_t credential_create(uid_t);

AUTH_WARN_RESULT AUTH_MALLOC AUTH_NONNULL_ALL AUTH_RETURNS_RETAINED    
credential_t credential_create_with_credential(credential_t,bool);
    
AUTH_WARN_RESULT AUTH_MALLOC AUTH_NONNULL_ALL AUTH_RETURNS_RETAINED    
credential_t credential_create_with_right(const char *);

AUTH_WARN_RESULT AUTH_MALLOC AUTH_RETURNS_RETAINED
credential_t credential_create_fvunlock(auth_items_t context, bool session);

AUTH_NONNULL_ALL
uid_t credential_get_uid(credential_t);

AUTH_NONNULL_ALL
const char * credential_get_name(credential_t);

AUTH_NONNULL_ALL
const char * credential_get_realname(credential_t);
    
AUTH_NONNULL_ALL
CFAbsoluteTime credential_get_creation_time(credential_t);
    
AUTH_NONNULL_ALL
bool credential_get_valid(credential_t);

AUTH_NONNULL_ALL    
bool credential_get_shared(credential_t);
    
AUTH_NONNULL_ALL
bool credential_is_right(credential_t);

AUTH_NONNULL_ALL    
bool credential_check_membership(credential_t,const char*);
    
AUTH_NONNULL_ALL
void credential_invalidate(credential_t);
    
#if defined(__cplusplus)
}
#endif

#endif /* !_SECURITY_AUTH_CREDENTIAL_H_ */
