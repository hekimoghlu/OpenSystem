/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 17, 2024.
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
//  dummy.h
//  MITKerberosShim
//
//  Created by Love HÃ¶rnquist Ã…strand on 2018-03-03.
//

#ifndef dummy_h
#define dummy_h

#include "gssapi.h"

void krb5int_accessor(void);
void krb5int_freeaddrinfo(void);
void krb5int_gai_strerror(void);
void krb5int_getaddrinfo(void);
void krb5int_gmt_mktime(void);
void krb5int_init_context_kdc(void);
void krb5int_pkinit_auth_pack_decode(void);
void krb5int_pkinit_create_cms_msg(void);
void krb5int_pkinit_pa_pk_as_rep_encode(void);
void krb5int_pkinit_pa_pk_as_req_decode(void);
void krb5int_pkinit_parse_cms_msg(void);
void krb5int_pkinit_reply_key_pack_encode(void);

void FSp_profile_init(void);
void FSp_profile_init_path(void);
void __KLSetApplicationPrompter(void);
void __KerberosDebugLogLevel(void);
void __KerberosDebugPrint(void);
void __KerberosDebugPrintMemory(void);
void __KerberosDebugPrintSession(void);
void __KerberosDebugVAPrint(void);
void __KerberosInternal_krb5_defkeyname(void);
void __KerberosInternal_krb5int_sendtokdc_debug_handler(void);
void encode_krb5_as_req(void);
void kim_options_create_from_stream(void);
void kim_options_write_to_stream(void);
void kim_selection_hints_create_from_stream(void);
void krb524_convert_creds_kdc(void);
void krb5_auth_con_getlocalsubkey(void);
void krb5_auth_con_getremotesubkey(void);
void krb5_auth_con_initivector(void);
void krb5_build_principal_va(void);
void krb5_free_config_files(void);
void krb5_free_enc_tkt_part(void);
void krb5_free_krbhst(void);
void krb5_get_default_config_files(void);
void krb5_get_in_tkt(void);
void krb5_get_in_tkt_with_keytab(void);
void krb5_get_in_tkt_with_password(void);
void krb5_get_in_tkt_with_skey(void);
void krb5_get_krbhst(void);
void krb5_get_realm_domain(void);
void krb5_gss_use_kdc_context(void);
void krb5_set_default_tgs_ktypes(void);

extern int krb5_use_broken_arcfour_string2key;

extern const gss_OID GSS_KRB5_MECHANISM;
extern const gss_OID GSS_KRB5_NT_MACHINE_UID_NAME;
extern const gss_OID GSS_KRB5_NT_STRING_UID_NAME;
extern const gss_OID GSS_KRB5_NT_USER_NAME;


#endif

