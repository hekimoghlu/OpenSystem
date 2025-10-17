/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 1, 2024.
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
#include "ntlm.h"

OM_uint32 _gss_ntlm_display_name
           (OM_uint32 * minor_status,
            const gss_name_t input_name,
            gss_buffer_t output_name_buffer,
            gss_OID * output_name_type
           )
{
    *minor_status = 0;

    if (output_name_type)
	*output_name_type = GSS_NTLM_MECHANISM;

    if (output_name_buffer) {
	ntlm_name n = (ntlm_name)input_name;
	char *str = NULL;
	int len;
	
	output_name_buffer->length = 0;
	output_name_buffer->value = NULL;

	if (n == NULL) {
	    *minor_status = 0;
	    return GSS_S_BAD_NAME;
	}

	len = asprintf(&str, "%s@%s", n->user, n->domain);
	if (len < 0 || str == NULL) {
	    *minor_status = ENOMEM;
	    return GSS_S_FAILURE;
	}
	output_name_buffer->length = len;
	output_name_buffer->value = str;
    }
    return GSS_S_COMPLETE;
}
