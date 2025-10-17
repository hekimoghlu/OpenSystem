/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 1, 2022.
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
#ifndef _SECURITY_AUTH_AUTHDB_H_
#define _SECURITY_AUTH_AUTHDB_H_

#include <sqlite3.h>
#include <CoreFoundation/CoreFoundation.h>

enum {
    AuthDBTransactionNone = 0,
    AuthDBTransactionImmediate,
    AuthDBTransactionExclusive,
    AuthDBTransactionNormal
};
typedef uint32_t AuthDBTransactionType;

#if defined(__cplusplus)
extern "C" {
#endif

#pragma mark -
#pragma mark authdb_t
    
typedef bool (^authdb_iterator_t)(auth_items_t data);
    
AUTH_WARN_RESULT AUTH_NONNULL_ALL
char * authdb_copy_sql_string(sqlite3_stmt*,int32_t);

AUTH_WARN_RESULT AUTH_MALLOC AUTH_RETURNS_RETAINED
authdb_t authdb_create(bool force_memory);

AUTH_WARN_RESULT AUTH_NONNULL_ALL
authdb_connection_t authdb_connection_acquire(authdb_t);

AUTH_NONNULL_ALL
void authdb_connection_release(authdb_connection_t*);
    
AUTH_NONNULL_ALL
bool authdb_maintenance(authdb_connection_t);
        
AUTH_NONNULL_ALL
int32_t authdb_exec(authdb_connection_t, const char *);

AUTH_NONNULL_ALL    
bool authdb_transaction(authdb_connection_t, AuthDBTransactionType, bool (^t)(void));
    
AUTH_NONNULL1 AUTH_NONNULL2
bool authdb_step(authdb_connection_t, const char * sql, void (^bind_stmt)(sqlite3_stmt* stmt), authdb_iterator_t iter);

AUTH_NONNULL_ALL    
int32_t authdb_get_key_value(authdb_connection_t, const char * table, const bool skip_maintenance, auth_items_t * out_items);

AUTH_NONNULL_ALL    
int32_t authdb_set_key_value(authdb_connection_t, const char * table, auth_items_t items);
    
AUTH_NONNULL_ALL
void authdb_checkpoint(authdb_connection_t);

AUTH_NONNULL1 AUTH_NONNULL2
bool authdb_import_plist(authdb_connection_t, CFDictionaryRef, bool, const char * name);

AUTH_NONNULL_ALL
OSStatus authdb_reset(authdb_connection_t);

#pragma mark -
#pragma mark authdb_connection_t
    
AUTH_NONNULL_ALL
authdb_connection_t authdb_connection_create(authdb_t);

AUTH_NONNULL_ALL
void authdb_check_for_mandatory_rights(authdb_connection_t);
    
#if defined(__cplusplus)
}
#endif

#endif /* !_SECURITY_AUTH_AUTHDB_H_ */
