/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 25, 2023.
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
#include "gssdigest.h"

static OM_uint32
scram_name(OM_uint32 *minor_status,
	   gss_const_OID mech,
	   const gss_buffer_t input_name_buffer,
	   gss_const_OID input_name_type,
	   gss_name_t *output_name)
{
    char *n = malloc(input_name_buffer->length + 1);
    if (n == NULL)
	return GSS_S_FAILURE;

    memcpy(n, input_name_buffer->value, input_name_buffer->length);
    n[input_name_buffer->length] = '\0';

    *output_name = (gss_name_t)n;

    return GSS_S_COMPLETE;
}

static struct _gss_name_type scram_names[] = {
    { GSS_C_NT_HOSTBASED_SERVICE, scram_name},
    { GSS_C_NT_USER_NAME, scram_name },
    { GSS_C_NT_EXPORT_NAME, scram_name },
    { NULL }
};


OM_uint32 _gss_scram_import_name
           (OM_uint32 * minor_status,
            const gss_buffer_t input_name_buffer,
            gss_const_OID input_name_type,
            gss_name_t * output_name
           )
{
    return _gss_mech_import_name(minor_status, GSS_SCRAM_MECHANISM,
				 scram_names, input_name_buffer,
				 input_name_type, output_name);
}

OM_uint32 _gss_scram_inquire_names_for_mech (
            OM_uint32 * minor_status,
            gss_const_OID mechanism,
            gss_OID_set * name_types
           )
{
    return _gss_mech_inquire_names_for_mech(minor_status, scram_names,
					    name_types);
}
