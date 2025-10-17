/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 24, 2022.
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
#include <gssapi_mech.h>

struct _gss_mech_switch {
	HEIM_SLIST_ENTRY(_gss_mech_switch)	gm_link;
	gss_OID_desc			gm_mech_oid;
	gss_OID_set			gm_name_types;
	void				*gm_so;
	gssapi_mech_interface_desc	gm_mech;
};
HEIM_SLIST_HEAD(_gss_mech_switch_list, _gss_mech_switch);
extern struct _gss_mech_switch_list _gss_mechs;
extern gss_OID_set _gss_mech_oids;

void	_gss_load_mech(void);
