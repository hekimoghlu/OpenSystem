/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 5, 2025.
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
#include "gsskrb5_locl.h"
#include <gssapi_mech.h>

/*
 * Context for krb5 calls.
 */

static gss_mo_desc krb5_mo[] = {
    {
	GSS_C_MA_SASL_MECH_NAME,
	GSS_MO_MA,
	"SASL mech name",
	rk_UNCONST("GS2-KRB5"),
	_gss_mo_get_ctx_as_string,
	NULL
    },
    {
	GSS_C_MA_MECH_NAME,
	GSS_MO_MA,
	"Mechanism name",
	rk_UNCONST("KRB5"),
	_gss_mo_get_ctx_as_string,
	NULL
    },
    {
	GSS_C_MA_MECH_DESCRIPTION,
	GSS_MO_MA,
	"Mechanism description",
	rk_UNCONST("Heimdal Kerberos 5 mech"),
	_gss_mo_get_ctx_as_string,
	NULL
    },
    {
	GSS_C_MA_MECH_CONCRETE,
	GSS_MO_MA
    },
    {
	GSS_C_MA_ITOK_FRAMED,
	GSS_MO_MA
    },
    {
	GSS_C_MA_AUTH_INIT,
	GSS_MO_MA
    },
    {
	GSS_C_MA_AUTH_TARG,
	GSS_MO_MA
    },
    {
	GSS_C_MA_AUTH_INIT_ANON,
	GSS_MO_MA
    },
    {
	GSS_C_MA_DELEG_CRED,
	GSS_MO_MA
    },
    {
	GSS_C_MA_INTEG_PROT,
	GSS_MO_MA
    },
    {
	GSS_C_MA_CONF_PROT,
	GSS_MO_MA
    },
    {
	GSS_C_MA_MIC,
	GSS_MO_MA
    },
    {
	GSS_C_MA_WRAP,
	GSS_MO_MA
    },
    {
	GSS_C_MA_PROT_READY,
	GSS_MO_MA
    },
    {
	GSS_C_MA_REPLAY_DET,
	GSS_MO_MA
    },
    {
	GSS_C_MA_OOS_DET,
	GSS_MO_MA
    },
    {
	GSS_C_MA_CBINDINGS,
	GSS_MO_MA
    },
    {
	GSS_C_MA_PFS,
	GSS_MO_MA
    },
    {
	GSS_C_MA_CTX_TRANS,
	GSS_MO_MA
    }
};

/*
 *
 */

static gssapi_mech_interface_desc krb5_mech = {
    GMI_VERSION,
    "kerberos 5",
    {9, rk_UNCONST("\x2a\x86\x48\x86\xf7\x12\x01\x02\x02") },
    0,
    _gsskrb5_acquire_cred,
    _gsskrb5_release_cred,
    _gsskrb5_init_sec_context,
    _gsskrb5_accept_sec_context,
    _gsskrb5_process_context_token,
    _gsskrb5_delete_sec_context,
    _gsskrb5_context_time,
    _gsskrb5_get_mic,
    _gsskrb5_verify_mic,
    _gsskrb5_wrap,
    _gsskrb5_unwrap,
    _gsskrb5_display_status,
    NULL,
    _gsskrb5_compare_name,
    _gsskrb5_display_name,
    _gsskrb5_import_name,
    _gsskrb5_export_name,
    _gsskrb5_release_name,
    _gsskrb5_inquire_cred,
    _gsskrb5_inquire_context,
    _gsskrb5_wrap_size_limit,
    _gsskrb5_add_cred,
    _gsskrb5_inquire_cred_by_mech,
    _gsskrb5_export_sec_context,
    _gsskrb5_import_sec_context,
    _gsskrb5_inquire_names_for_mech,
    _gsskrb5_inquire_mechs_for_name,
    _gsskrb5_canonicalize_name,
    _gsskrb5_duplicate_name,
    _gsskrb5_inquire_sec_context_by_oid,
    _gsskrb5_inquire_cred_by_oid,
    _gsskrb5_set_sec_context_option,
    _gsskrb5_set_cred_option,
    _gsskrb5_pseudo_random,
    _gk_wrap_iov,
    _gk_unwrap_iov,
    _gk_wrap_iov_length,
    _gsskrb5_store_cred,
    _gsskrb5_export_cred,
    _gsskrb5_import_cred,
    _gss_krb5_acquire_cred_ext,
    _gss_krb5_iter_creds_f,
    _gsskrb5_destroy_cred,
    _gsskrb5_cred_hold,
    _gsskrb5_cred_unhold,
    _gsskrb5_cred_label_get,
    _gsskrb5_cred_label_set,
    krb5_mo,
    sizeof(krb5_mo) / sizeof(krb5_mo[0]),
    _gsskrb5_pname_to_uid,
    _gsskrb5_authorize_localname,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    _gsskrb5_appl_change_password
};

static gssapi_mech_interface_desc iakerb_mech = {
    GMI_VERSION,
    "iakerb",
    {6, "\x2b\x06\x01\x05\x02\x05" },
    0,
    _gssiakerb_acquire_cred,
    _gsskrb5_release_cred,
    _gsskrb5_init_sec_context,
    _gssiakerb_accept_sec_context,
    _gsskrb5_process_context_token,
    _gsskrb5_delete_sec_context,
    _gsskrb5_context_time,
    _gsskrb5_get_mic,
    _gsskrb5_verify_mic,
    _gsskrb5_wrap,
    _gsskrb5_unwrap,
    _gsskrb5_display_status,
    NULL,
    _gsskrb5_compare_name,
    _gsskrb5_display_name,
    _gssiakerb_import_name,
    _gssiakerb_export_name,
    _gsskrb5_release_name,
    _gsskrb5_inquire_cred,
    _gsskrb5_inquire_context,
    _gsskrb5_wrap_size_limit,
    _gsskrb5_add_cred,
    _gsskrb5_inquire_cred_by_mech,
    _gsskrb5_export_sec_context,
    _gsskrb5_import_sec_context,
    _gssiakerb_inquire_names_for_mech,
    _gsskrb5_inquire_mechs_for_name,
    _gsskrb5_canonicalize_name,
    _gsskrb5_duplicate_name,
    _gsskrb5_inquire_sec_context_by_oid,
    _gsskrb5_inquire_cred_by_oid,
    _gsskrb5_set_sec_context_option,
    _gsskrb5_set_cred_option,
    _gsskrb5_pseudo_random,
    _gk_wrap_iov,
    _gk_unwrap_iov,
    _gk_wrap_iov_length,
    _gsskrb5_store_cred,
    _gsskrb5_export_cred,
    _gsskrb5_import_cred,
    _gss_iakerb_acquire_cred_ext,
    _gss_iakerb_iter_creds_f,
    _gsskrb5_destroy_cred,
    _gsskrb5_cred_hold,
    _gsskrb5_cred_unhold,
    _gsskrb5_cred_label_get,
    _gsskrb5_cred_label_set,
    NULL,
    0,
    _gsskrb5_pname_to_uid,
    _gsskrb5_authorize_localname,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    _gsskrb5_appl_change_password
};


#ifdef PKINIT

static gssapi_mech_interface_desc pku2u_mech = {
    GMI_VERSION,
    "pku2u",
    {6, "\x2b\x05\x01\x05\x02\x07" },
    0,
    _gsspku2u_acquire_cred,
    _gsskrb5_release_cred,
    _gsskrb5_init_sec_context,
    _gsspku2u_accept_sec_context,
    _gsskrb5_process_context_token,
    _gsskrb5_delete_sec_context,
    _gsskrb5_context_time,
    _gsskrb5_get_mic,
    _gsskrb5_verify_mic,
    _gsskrb5_wrap,
    _gsskrb5_unwrap,
    _gsskrb5_display_status,
    NULL,
    _gsskrb5_compare_name,
    _gsskrb5_display_name,
    _gsspku2u_import_name,
    _gsspku2u_export_name,
    _gsskrb5_release_name,
    _gsskrb5_inquire_cred,
    _gsskrb5_inquire_context,
    _gsskrb5_wrap_size_limit,
    _gsskrb5_add_cred,
    _gsskrb5_inquire_cred_by_mech,
    _gsskrb5_export_sec_context,
    _gsskrb5_import_sec_context,
    _gsspku2u_inquire_names_for_mech,
    _gsskrb5_inquire_mechs_for_name,
    _gsskrb5_canonicalize_name,
    _gsskrb5_duplicate_name,
    _gsskrb5_inquire_sec_context_by_oid,
    _gsskrb5_inquire_cred_by_oid,
    _gsskrb5_set_sec_context_option,
    _gsskrb5_set_cred_option,
    _gsskrb5_pseudo_random,
    _gk_wrap_iov,
    _gk_unwrap_iov,
    _gk_wrap_iov_length,
    _gsskrb5_store_cred,
    _gsskrb5_export_cred,
    _gsskrb5_import_cred,
    _gss_krb5_acquire_cred_ext,
    _gss_pku2u_iter_creds_f,
    _gsskrb5_destroy_cred,
    _gsskrb5_cred_hold,
    _gsskrb5_cred_unhold,
    _gsskrb5_cred_label_get,
    _gsskrb5_cred_label_set,
    NULL,
    0,
    _gsskrb5_pname_to_uid,
    _gsskrb5_authorize_localname,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    _gsskrb5_appl_change_password
};

#endif

gssapi_mech_interface
__gss_krb5_initialize(void)
{
    return &krb5_mech;
}

gssapi_mech_interface
__gss_pku2u_initialize(void)
{
    return &iakerb_mech;
}

gssapi_mech_interface
__gss_iakerb_initialize(void)
{
#ifdef PKINIT
    return &pku2u_mech;
#else
    return NULL;
#endif
}

/*
 * compat glue
 */

gss_OID_desc GSSAPI_LIB_VARIABLE __gss_appl_lkdc_supported_desc =
    {6, rk_UNCONST("\x2a\x85\x70\x2b\x0e\x03") };
gss_OID_desc GSSAPI_LIB_VARIABLE __gss_c_nt_uuid_desc =
    {6, rk_UNCONST("\x2a\x85\x70\x2b\x0d\x1e")};
