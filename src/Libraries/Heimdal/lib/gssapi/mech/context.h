/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 2, 2025.
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

struct _gss_context {
	gssapi_mech_interface	gc_mech;
	gss_ctx_id_t		gc_ctx;
	gss_cred_id_t		gc_replaced_cred;
};

void
_gss_mg_error(__nonnull gssapi_mech_interface, OM_uint32);

OM_uint32
_gss_mg_get_error(__nonnull const gss_OID, OM_uint32, __nonnull gss_buffer_t);

void
_gss_load_plugins(void);
