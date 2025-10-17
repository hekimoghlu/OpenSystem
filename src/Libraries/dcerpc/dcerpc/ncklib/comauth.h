/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 25, 2023.
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
**
**  NAME
**
**      comauth.h
**
**  FACILITY:
**
**      Remote Procedure Call (RPC)
**
**  ABSTRACT:
**
**  Generic interface to authentication services
**
**
*/

#ifndef _COMAUTH_H
#define _COMAUTH_H	1

/***********************************************************************/
/*
 * Function pointer types of functions exported by an authentication
 * service.
 */

/*
 * The next four function pointer types correspond to the similarly
 * named functions in the API.
 */

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*rpc_auth_bnd_set_auth_info_fn_t) (
        unsigned_char_p_t                   /* in  */   /*server_princ_name*/,
        rpc_authn_level_t                   /* in  */   /*authn_level*/,
        rpc_auth_identity_handle_t          /* in  */   /*auth_identity*/,
        rpc_authz_protocol_id_t             /* in  */   /*authz_protocol*/,
        rpc_binding_handle_t                /* in  */   /*binding_h*/,
        rpc_auth_info_p_t                   /* out*/   * /*auth_info*/,
        unsigned32                          /* out */   *st
    );

typedef void (*rpc_auth_serv_reg_auth_info_fn_t) (
        unsigned_char_p_t                   /* in  */   /*server_princ_name*/,
        rpc_auth_key_retrieval_fn_t         /* in  */   /*get_key_func*/,
        dce_pointer_t                           /* in  */   /*arg*/,
        unsigned32                          /* out */   *st
    );

typedef void (*rpc_mgmt_inq_def_authn_lvl_fn_t) (
        unsigned32                          /* out */   * /*authn_level*/,
        unsigned32                          /* out */   *st
    );

/*
 * This function pointer type describes an authentication
 * procedure that frees the auth info memory and resources.
 */

typedef void (*rpc_auth_free_info_fn_t) (
        rpc_auth_info_p_t                   /* in/out */  * /*auth_info*/
    );

/*
 * This function pointer type defines a procedure which frees key info
 * memory and resources.
 */

typedef void (*rpc_auth_free_key_fn_t) (
        rpc_key_info_p_t                    /* in/out */ * /*key_info*/
    );

/*
 * These function pointer types resolve a passed-in "auth_identity"
 * into a reference to a "real" auth identity, and release the
 * reference.  Note that these return the error status in the return
 * value (instead of in an output argument) because this means that
 * the function pointer has the exact same signature as the routine in
 * the security component which implements this function.
 */

typedef error_status_t (*rpc_auth_resolve_identity_fn_t) (
        rpc_auth_identity_handle_t                 /* in */ /*in_identity*/,
        rpc_auth_identity_handle_t                 /* out */ *out_identity
    );

typedef void (*rpc_auth_release_identity_fn_t) (
        rpc_auth_identity_handle_t                 /* in/out */ * /*identity*/
    );

/*
 * This function pointer type describes an authentication service procedure
 * that returns the principal name (possibly just one of many) that server
 * is running as.
 */

typedef void (*rpc_auth_inq_my_princ_name_fn_t) (
        unsigned32                          /* in */    /*princ_name_size*/,
        unsigned_char_p_t                   /* out */   /*princ_name*/,
        unsigned32                          /* out */   * /*st*/
    );

/*
 * This function pointer type describes an authentication service procedure
 * that returns the mechanism-specific security context.
 */

typedef void (*rpc_auth_inq_sec_context_fn_t) (
        rpc_auth_info_p_t                   /* in */    /*auth_info*/,
        void                                /* out */   ** /*mech_context*/,
        unsigned32                          /* out */   * /*st*/
   );

typedef void (*rpc_auth_inq_access_token_fn_t) (
        rpc_auth_info_p_t,
        rpc_access_token_p_t*,
        unsigned32*);

/***********************************************************************/
/*
 * RPC authentication service API EPV.
 */

typedef struct
{
    rpc_auth_bnd_set_auth_info_fn_t     binding_set_auth_info;
    rpc_auth_serv_reg_auth_info_fn_t    server_register_auth_info;
    rpc_mgmt_inq_def_authn_lvl_fn_t     mgmt_inq_dflt_auth_level;
    rpc_auth_inq_my_princ_name_fn_t     inq_my_princ_name;
    rpc_auth_free_info_fn_t             free_info;
    rpc_auth_free_key_fn_t              free_key;
    rpc_auth_resolve_identity_fn_t      resolve_id;
    rpc_auth_release_identity_fn_t      release_id;
    rpc_auth_inq_sec_context_fn_t       inq_sec_context;
    rpc_auth_inq_access_token_fn_t      inq_access_token;
} rpc_auth_epv_t, *rpc_auth_epv_p_t;

/***********************************************************************/

PRIVATE void rpc__auth_info_binding_release (
        rpc_binding_rep_p_t     binding_rep
    );

PRIVATE void rpc__np_auth_info_binding_release (
        rpc_binding_rep_p_t     binding_rep
    );

PRIVATE void rpc__auth_inq_my_princ_name (
        unsigned32              /*dce_rpc_authn_protocol*/,
        unsigned32              /*princ_name_size*/,
        unsigned_char_p_t       /*princ_name*/,
        unsigned32              * /*status*/
    );

PRIVATE void rpc__auth_info_cache_init (
        unsigned32              * /*status*/
    );

/***********************************************************************/

/*
 * Signature of the init routine provided.  Each authentication service
 * must return both an API and a common EPV.
 */

typedef void (*rpc_auth_init_fn_t) (
        rpc_auth_epv_p_t            * /*auth_epv*/,
        rpc_auth_rpc_prot_epv_tbl_t * /*auth_rpc_prot_epv_tbl*/,
        unsigned32                  * /*status*/
    );

/*
 * Declarations of the RPC authentication service init routines.
 */

void rpc__krb_init (
        rpc_auth_epv_p_t            * /*auth_epv*/,
        rpc_auth_rpc_prot_epv_tbl_t * /*auth_rpc_prot_epv_tbl*/,
        unsigned32                  * /*status*/
    );

void rpc__noauth_init (
        rpc_auth_epv_p_t            * /*auth_epv*/,
        rpc_auth_rpc_prot_epv_tbl_t * /*auth_rpc_prot_epv_tbl*/,
        unsigned32                  * /*status*/
    );

void rpc__key_info_release (
    rpc_key_info_p_t *
);

void rpc__key_info_reference (
    rpc_key_info_p_t
);

#ifdef _cplusplus
}
#endif

#endif /*  _COMAUTH_H */
