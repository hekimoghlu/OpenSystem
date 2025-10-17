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
#import <stdio.h>

#import <Foundation/Foundation.h>
#import <Heimdal/krb5.h>
#import <GSS/gssapi_spi.h>
#import <GSS/gssapi_plugin.h>

#include <syslog.h>
#include <assert.h>

/**
 * Icky code that prints the target name and replace the cred with another cred that hard coded in the module
 */

#define GSSC_MAGIC 0x47111147

struct gsssel_ctx {
    int magic;
};

static const char selectuser[] = "lha@KTH.SE";
static const char *replacenames[] = {
    "host@svn.h5l.org", 
    NULL
};

static gss_cred_id_t
isc_replace_cred(gss_name_t target, gss_OID mech, gss_cred_id_t original_cred, OM_uint32 flags)
{
    OM_uint32 maj_stat, min_stat;
    gss_buffer_desc buffer;
    gss_name_t name;
    gss_cred_id_t newcred;
    bool exchange = false;
    size_t n;

    /*
     * Using gss_display_name() is wrong, however it the best we can
     * do right now.
     *
     * We should use gss_import_name() and then compare the name with
     * gss_compare_name(), the only issue with that is that comparing
     * names fragile and do not work as expected wrt to case of
     * string, and in case or Kerberos, hostbased service gets
     * affected by realm.
     */
    
    maj_stat = gss_display_name(&min_stat, target, &buffer, NULL);
    if (maj_stat == GSS_S_COMPLETE) {
	syslog(LOG_ERR, "ISC-replace-cred target name: %.*s", (int)buffer.length, (char *)buffer.value);
	for (n = 0; replacenames[n]; n++) {
	    if (memmem(buffer.value, buffer.length, replacenames[n], strlen(replacenames[n])) != NULL)
		exchange = true;
	}
	gss_release_buffer(&min_stat, &buffer);
    }
    if (!exchange) {
	syslog(LOG_ERR, "ISC-replace-cred not replacing");
	return NULL;
    }
    
    buffer.value = (char *)(uintptr_t)selectuser;
    buffer.length = strlen(selectuser);
    maj_stat = gss_import_name(&min_stat, &buffer, GSS_C_NT_USER_NAME, &name);
    if (maj_stat != GSS_S_COMPLETE)
	return NULL;
    
    maj_stat = gss_acquire_cred(&min_stat, name, GSS_C_INDEFINITE, GSS_C_NO_OID_SET, GSS_C_INITIATE, 
				&newcred, NULL, NULL);
    gss_release_name(&min_stat, &name);
    if (maj_stat != GSS_S_COMPLETE)
	return NULL;
    
    syslog(LOG_ERR, "ISC-replace-cred replacing cred to: %s", selectuser);
    
    return newcred;
}

static krb5_error_code
gsssel_init(krb5_context context, void **ptr)
{
    struct gsssel_ctx *ctx = calloc(1, sizeof(*ctx));

    if (ctx == NULL)
	return ENOMEM;
    
    ctx->magic = GSSC_MAGIC;

    *ptr = ctx;
    return 0;
}

static void
gsssel_fini(void *ptr)
{
    struct gsssel_ctx *ctx = ptr;
    
    assert(ctx->magic == GSSC_MAGIC);

    free(ctx);
}


gssapi_plugin_ftable gssapi_plugin = {
    GSSAPI_PLUGIN_VERSION_1,
    gsssel_init,
    gsssel_fini,
    "gssapi credential selector",
    0,
    isc_replace_cred
};
