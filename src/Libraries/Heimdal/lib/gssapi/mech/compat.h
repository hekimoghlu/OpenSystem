/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 7, 2024.
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
typedef OM_uint32 GSSAPI_CALLCONV _gss_inquire_saslname_for_mech_t (
	       OM_uint32 *,           /* minor_status */
	       const gss_OID,         /* desired_mech */
	       gss_buffer_t,          /* sasl_mech_name */
	       gss_buffer_t,          /* mech_name */
	       gss_buffer_t           /* mech_description */
	    );

typedef OM_uint32 GSSAPI_CALLCONV _gss_inquire_mech_for_saslname_t (
	       OM_uint32 *,           /* minor_status */
	       const gss_buffer_t,    /* sasl_mech_name */
	       gss_OID *              /* mech_type */
	    );

typedef OM_uint32 GSSAPI_CALLCONV _gss_inquire_attrs_for_mech_t (
	       OM_uint32 *,           /* minor_status */
	       gss_const_OID,         /* mech */
	       gss_OID_set *,         /* mech_attrs */
	       gss_OID_set *          /* known_mech_attrs */
	    );

typedef OM_uint32 GSSAPI_CALLCONV _gss_acquire_cred_with_password_t
	      (OM_uint32 *,            /* minor_status */
	       const gss_name_t,       /* desired_name */
	       const gss_buffer_t,     /* password */
	       OM_uint32,              /* time_req */
	       const gss_OID_set,      /* desired_mechs */
	       gss_cred_usage_t,       /* cred_usage */
	       gss_cred_id_t *,        /* output_cred_handle */
	       gss_OID_set *,          /* actual_mechs */
	       OM_uint32 *             /* time_rec */
	      );

typedef OM_uint32 GSSAPI_CALLCONV _gss_add_cred_with_password_t (
	       OM_uint32 *,            /* minor_status */
	       const gss_cred_id_t,    /* input_cred_handle */
	       const gss_name_t,       /* desired_name */
	       const gss_OID,          /* desired_mech */
	       const gss_buffer_t,     /* password */
	       gss_cred_usage_t,       /* cred_usage */
	       OM_uint32,              /* initiator_time_req */
	       OM_uint32,              /* acceptor_time_req */
	       gss_cred_id_t *,        /* output_cred_handle */
	       gss_OID_set *,          /* actual_mechs */
	       OM_uint32 *,            /* initiator_time_rec */
	       OM_uint32 *             /* acceptor_time_rec */
	      );

/*
 * API-as-SPI compatibility for compatibility with MIT mechanisms;
 * native Heimdal mechanisms should not use these.
 */
struct gss_mech_compat_desc_struct {
	_gss_inquire_saslname_for_mech_t    *gmc_inquire_saslname_for_mech;
	_gss_inquire_mech_for_saslname_t    *gmc_inquire_mech_for_saslname;
	_gss_inquire_attrs_for_mech_t       *gmc_inquire_attrs_for_mech;
	_gss_acquire_cred_with_password_t   *gmc_acquire_cred_with_password;
#if 0
	_gss_add_cred_with_password_t       *gmc_add_cred_with_password;
#endif
};

