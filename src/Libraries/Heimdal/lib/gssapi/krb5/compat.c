/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 27, 2024.
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

static krb5_error_code
check_compat(OM_uint32 *minor_status,
	     krb5_context context, krb5_const_principal name,
	     const char *option, krb5_boolean *compat,
	     krb5_boolean match_val)
{
    krb5_error_code ret = 0;
    char **p, **q;
    krb5_principal match;


    p = krb5_config_get_strings(context, NULL, "gssapi",
				option, NULL);
    if(p == NULL)
	return 0;

    match = NULL;
    for(q = p; *q; q++) {
	ret = krb5_parse_name(context, *q, &match);
	if (ret)
	    break;

	if (krb5_principal_match(context, name, match)) {
	    *compat = match_val;
	    break;
	}

	krb5_free_principal(context, match);
	match = NULL;
    }
    if (match)
	krb5_free_principal(context, match);
    krb5_config_free_strings(p);

    if (ret) {
	if (minor_status)
	    *minor_status = ret;
	return GSS_S_FAILURE;
    }

    return 0;
}

/*
 * ctx->ctx_id_mutex is assumed to be locked
 */

OM_uint32
_gss_DES3_get_mic_compat(OM_uint32 *minor_status,
			 gsskrb5_ctx ctx,
			 krb5_context context)
{
    krb5_boolean use_compat = FALSE;
    OM_uint32 ret;

    if ((ctx->more_flags & COMPAT_OLD_DES3_SELECTED) == 0) {
	ret = check_compat(minor_status, context, ctx->target,
			   "broken_des3_mic", &use_compat, TRUE);
	if (ret)
	    return ret;
	ret = check_compat(minor_status, context, ctx->target,
			   "correct_des3_mic", &use_compat, FALSE);
	if (ret)
	    return ret;

	if (use_compat)
	    ctx->more_flags |= COMPAT_OLD_DES3;
	ctx->more_flags |= COMPAT_OLD_DES3_SELECTED;
    }
    return 0;
}

#if 0
OM_uint32
gss_krb5_compat_des3_mic(OM_uint32 *minor_status, gss_ctx_id_t ctx, int on)
{
    *minor_status = 0;

    HEIMDAL_MUTEX_lock(&ctx->ctx_id_mutex);
    if (on) {
	ctx->more_flags |= COMPAT_OLD_DES3;
    } else {
	ctx->more_flags &= ~COMPAT_OLD_DES3;
    }
    ctx->more_flags |= COMPAT_OLD_DES3_SELECTED;
    HEIMDAL_MUTEX_unlock(&ctx->ctx_id_mutex);

    return 0;
}
#endif
