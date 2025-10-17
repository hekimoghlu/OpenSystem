/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 16, 2024.
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
struct _gss_mechanism_cred {
	HEIM_SLIST_ENTRY(_gss_mechanism_cred) gmc_link;
	gssapi_mech_interface	gmc_mech;	/* mechanism ops for MC */
	gss_OID			gmc_mech_oid;	/* mechanism oid for MC */
	gss_cred_id_t		gmc_cred;	/* underlying MC */
};
HEIM_SLIST_HEAD(_gss_mechanism_cred_list, _gss_mechanism_cred);

struct _gss_cred {
	struct _gss_mechanism_cred_list gc_mc;
};

struct _gss_cred *
_gss_mg_alloc_cred(void);

void
_gss_mg_release_cred(struct _gss_cred *cred);

struct _gss_mechanism_cred *
_gss_copy_cred(struct _gss_mechanism_cred *mc);

struct _gss_mechanism_name;

OM_uint32
_gss_acquire_mech_cred(OM_uint32 *minor_status,
		       struct gssapi_mech_interface_desc *m,
		       const struct _gss_mechanism_name *mn,
		       gss_const_OID credential_type,
		       const void *credential_data,
		       OM_uint32 time_req,
		       gss_const_OID desired_mech,
		       gss_cred_usage_t cred_usage,
		       struct _gss_mechanism_cred **output_cred_handle);

