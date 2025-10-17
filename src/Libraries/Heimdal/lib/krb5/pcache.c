/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 30, 2024.
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
#include "krb5_locl.h"
#include "ccache_plugin.h"
#ifdef HAVE_DLFCN_H
#include <dlfcn.h>
#endif
#include <assert.h>

static krb5_error_code KRB5_LIB_CALL
callback(krb5_context context, const void *plug, void *plugctx, void *userctx)
{
    krb5_cc_ops *ccops = (krb5_cc_ops *)plug;
    krb5_error_code ret;

    if (ccops != NULL && ccops->version >= KRB5_CC_OPS_VERSION)
	return KRB5_PLUGIN_NO_HANDLE;

    ret = krb5_cc_register(context, ccops, TRUE);
    if (ret != 0)
	*((krb5_error_code *)userctx) = ret;

    return KRB5_PLUGIN_NO_HANDLE;
}


krb5_error_code
_krb5_load_ccache_plugins(krb5_context context)
{
    krb5_error_code userctx = 0;

    (void)krb5_plugin_run_f(context, "krb5", KRB5_PLUGIN_CCACHE,
			    0, 0, &userctx, callback);

    return userctx;
}
