/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 25, 2022.
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
/*
 * Header file for red black tree of the ldap response messages.
 */
 

#ifndef _RB_RESPONSE_H_
#define _RB_RESPONSE_H_
 
#ifdef LDAP_RESPONSE_RB_TREE

#include "ldap.h"
#include "ldap-int.h"
#include "lber_types.h"
 
void ldap_resp_rbt_create( LDAP *ld );
void ldap_resp_rbt_free( LDAP *ld );
void ldap_resp_rbt_insert_msg( LDAP *ld, LDAPMessage *lm );
void ldap_resp_rbt_delete_msg( LDAP *ld, LDAPMessage *lm );
void ldap_resp_rbt_unlink_msg( LDAP *ld, LDAPMessage *lm);
void ldap_resp_rbt_unlink_partial_msg( LDAP *ld, LDAPMessage *lm );
LDAPMessage *ldap_resp_rbt_find_msg( LDAP* ld, ber_int_t msgid );
LDAPMessage *ldap_resp_rbt_get_first_msg( LDAP *ld );
LDAPMessage *ldap_resp_rbt_get_next_msg( LDAP *ld, LDAPMessage *lm );
void ldap_resp_rbt_dump( LDAP *ld );

#endif /* LDAP_RESPONSE_RB_TREE */

#endif /* _RB_RESPONSE_H_ */
 