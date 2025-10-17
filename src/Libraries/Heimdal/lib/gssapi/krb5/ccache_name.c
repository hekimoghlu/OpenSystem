/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 11, 2022.
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
#include "gsskrb5_locl.h"

char *last_out_name;

OM_uint32
_gsskrb5_krb5_ccache_name(OM_uint32 *minor_status,
			  const char *name,
			  const char **out_name)
{
    krb5_context context;
    krb5_error_code kret;

    *minor_status = 0;

    GSSAPI_KRB5_INIT(&context);

    if (out_name) {
	const char *n;

	if (last_out_name) {
	    free(last_out_name);
	    last_out_name = NULL;
	}

	n = krb5_cc_default_name(context);
	if (n == NULL) {
	    *minor_status = ENOMEM;
	    return GSS_S_FAILURE;
	}
	last_out_name = strdup(n);
	if (last_out_name == NULL) {
	    *minor_status = ENOMEM;
	    return GSS_S_FAILURE;
	}
	*out_name = last_out_name;
    }

    kret = krb5_cc_set_default_name(context, name);
    if (kret) {
	*minor_status = kret;
	return GSS_S_FAILURE;
    }
    return GSS_S_COMPLETE;
}
