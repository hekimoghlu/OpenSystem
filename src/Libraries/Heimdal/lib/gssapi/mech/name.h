/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 30, 2023.
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
struct _gss_mechanism_name {
	HEIM_SLIST_ENTRY(_gss_mechanism_name) gmn_link;
	gssapi_mech_interface	gmn_mech;	/* mechanism ops for MN */
	gss_OID			gmn_mech_oid;	/* mechanism oid for MN */
	gss_name_t		gmn_name;	/* underlying MN */
};
HEIM_SLIST_HEAD(_gss_mechanism_name_list, _gss_mechanism_name);

struct _gss_name {
	gss_OID_desc		gn_type;	/* type of name */
	gss_buffer_desc		gn_value;	/* value (as imported) */
	struct _gss_mechanism_name_list gn_mn;	/* list of MNs */
};

OM_uint32
	_gss_find_mn(OM_uint32 *, struct _gss_name *, gss_const_OID,
	      struct _gss_mechanism_name **);
struct _gss_name *
	_gss_create_name(gss_name_t new_mn, gssapi_mech_interface m);
void	_gss_mg_release_name(struct _gss_name *);
