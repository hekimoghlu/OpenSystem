/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 3, 2023.
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
#include "netlogon.h"

OM_uint32 _netlogon_display_name
           (OM_uint32 * minor_status,
            const gss_name_t input_name,
            gss_buffer_t output_name_buffer,
            gss_OID * output_name_type
           )
{
    const gssnetlogon_name name = (const gssnetlogon_name)input_name;
    gss_buffer_t namebuf;

    if (output_name_type != NULL)
        *output_name_type = GSS_C_NO_OID;

    if (output_name_buffer != NULL) {
        namebuf = name->DnsName.length ? &name->DnsName : &name->NetbiosName;

        output_name_buffer->value = malloc(namebuf->length + 1);
        if (output_name_buffer->value == NULL) {
            *minor_status = ENOMEM;
            return GSS_S_FAILURE;
        }
        memcpy(output_name_buffer->value, namebuf->value, namebuf->length);
        ((char *)output_name_buffer->value)[namebuf->length] = '\0';
        output_name_buffer->length = namebuf->length;
    }

    *minor_status = 0;
    return GSS_S_COMPLETE;
}

