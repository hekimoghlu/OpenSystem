/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 21, 2024.
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
#include "spnego_locl.h"

OM_uint32 GSSAPI_CALLCONV
_gss_spnego_export_cred (OM_uint32 *minor_status,
			 gss_cred_id_t cred_handle,
			 gss_buffer_t value)
{
    return gss_export_cred(minor_status, cred_handle, value);
}

OM_uint32 GSSAPI_CALLCONV
_gss_spnego_import_cred (OM_uint32 *minor_status,
			 gss_buffer_t value,
			 gss_cred_id_t *cred_handle)
{
    return gss_import_cred(minor_status, value, cred_handle);
}

