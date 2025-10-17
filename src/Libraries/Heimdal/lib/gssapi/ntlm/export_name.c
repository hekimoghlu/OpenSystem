/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 16, 2021.
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

OM_uint32 _gss_ntlm_export_name
           (OM_uint32  * minor_status,
            const gss_name_t input_name,
            gss_buffer_t exported_name
           )
{
    OM_uint32 major_status;
    ntlm_name name = (ntlm_name)input_name;
    char *str = NULL;

    asprintf(&str, "%s\\%s", name->domain, name->user);
    if (str == NULL) {
	*minor_status = ENOMEM;
	return GSS_S_FAILURE;
    }

    major_status = gss_mg_export_name(minor_status, GSS_NTLM_MECHANISM,
				      str, strlen(str), exported_name);
    free(str);
    return major_status;
}
