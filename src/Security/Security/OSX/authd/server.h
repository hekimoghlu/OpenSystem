/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 19, 2023.
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
#ifndef _SECURITY_AUTH_SERVER_H_
#define _SECURITY_AUTH_SERVER_H_

#include "authd_private.h"
#include <xpc/xpc.h>
#include "Authorization.h"

#if defined(__cplusplus)
extern "C" {
#endif

OSStatus server_init(void);
void server_cleanup(void);
bool server_in_dark_wake(void);
authdb_t server_get_database(void);
    
AUTH_NONNULL_ALL
connection_t server_register_connection(xpc_connection_t);

AUTH_NONNULL_ALL
void server_unregister_connection(connection_t);
    
AUTH_NONNULL_ALL
void server_register_auth_token(auth_token_t);
 
AUTH_NONNULL_ALL
void server_unregister_auth_token(auth_token_t);

AUTH_NONNULL_ALL
auth_token_t server_find_copy_auth_token(AuthorizationBlob * blob);
    
AUTH_NONNULL_ALL
session_t server_find_copy_session(session_id_t,bool create);

void server_dev(void);
    
/* API */
    
AUTH_NONNULL_ALL
OSStatus authorization_create(connection_t,xpc_object_t,xpc_object_t);

AUTH_NONNULL_ALL
OSStatus authorization_create_with_audit_token(connection_t,xpc_object_t,xpc_object_t);
    
AUTH_NONNULL_ALL
OSStatus authorization_free(connection_t,xpc_object_t,xpc_object_t);
    
AUTH_NONNULL_ALL
OSStatus authorization_copy_right_properties(connection_t, xpc_object_t, xpc_object_t);

AUTH_NONNULL_ALL
OSStatus authorization_copy_rights(connection_t,xpc_object_t,xpc_object_t);
    
AUTH_NONNULL_ALL
OSStatus authorization_copy_info(connection_t,xpc_object_t,xpc_object_t);
    
AUTH_NONNULL_ALL
OSStatus authorization_make_external_form(connection_t,xpc_object_t,xpc_object_t);
    
AUTH_NONNULL_ALL
OSStatus authorization_create_from_external_form(connection_t,xpc_object_t,xpc_object_t);
    
AUTH_NONNULL_ALL
OSStatus authorization_right_get(connection_t,xpc_object_t,xpc_object_t);
    
AUTH_NONNULL_ALL
OSStatus authorization_right_set(connection_t,xpc_object_t,xpc_object_t);

AUTH_NONNULL_ALL
OSStatus authorization_right_remove(connection_t,xpc_object_t,xpc_object_t);
    
AUTH_NONNULL_ALL
OSStatus session_set_user_preferences(connection_t,xpc_object_t,xpc_object_t);
    
AUTH_NONNULL_ALL
OSStatus authorization_copy_prelogin_userdb(connection_t,xpc_object_t,xpc_object_t);

AUTH_NONNULL_ALL
OSStatus
authorization_copy_prelogin_pref_value(connection_t conn, xpc_object_t message, xpc_object_t reply);

AUTH_NONNULL_ALL
OSStatus
authorization_prelogin_smartcardonly_override(connection_t conn, xpc_object_t message, xpc_object_t reply);

AUTH_NONNULL2
OSStatus
server_authorize(connection_t, auth_token_t, AuthorizationFlags, auth_rights_t, auth_items_t, engine_t *);

#if defined(__cplusplus)
}
#endif

#endif /* !_SECURITY_AUTH_SERVER_H_ */
