/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 17, 2024.
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
#include "heim.h"
#include "mit-gssapi.h"
#include "mit-gssapi_krb5.h"
#include <string.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>

OM_uint32
heim_gsskrb5_extract_authz_data_from_sec_context(OM_uint32 * /*minor_status*/,
						 gss_ctx_id_t /*context_handle*/,
						 int /*ad_type*/,
						 gss_buffer_t /*ad_data*/);

uint32_t KRB5_CALLCONV
apple_gss_krb5_export_authdata_if_relevant_context(uint32_t *min_stat,
						   gss_ctx_id_t *context_handle,
						   uint32_t version,
						   void **kctx)
{
    apple_gss_krb5_authdata_if_relevant *d;
    gss_buffer_desc buffer;
    uint32_t maj_stat;

    if (version != 1 && *context_handle == NULL) {
	*min_stat = EINVAL;
	return GSS_S_FAILURE;
    }

    maj_stat = heim_gsskrb5_extract_authz_data_from_sec_context(min_stat,
								*context_handle,
								KRB5_AUTHDATA_IF_RELEVANT,
								&buffer);
    if (maj_stat)
	return maj_stat;

    d = calloc(1, sizeof(*d));
    if (d == NULL) {
	gss_release_buffer(min_stat, &buffer);
	return GSS_S_FAILURE;
    }

    d->type = KRB5_AUTHDATA_IF_RELEVANT;
    d->length = (OM_uint32)buffer.length;
    d->data = malloc(buffer.length);
    if (d->data == NULL) {
	gss_release_buffer(min_stat, &buffer);
	free(d);
	*min_stat = 0;
	return GSS_S_FAILURE;
    }
    memcpy(d->data, buffer.value, buffer.length);

    gss_release_buffer(min_stat, &buffer);

    *kctx = d;

    *min_stat = 0;
    return GSS_S_COMPLETE;
}

uint32_t
apple_gss_krb5_free_authdata_if_relevant(uint32_t *minor_status,
					 void *kctx)
{
    apple_gss_krb5_authdata_if_relevant *d = kctx;
    
    if (d) {
	if (d->data)
	    free(d->data);
	free(d);
    }
    *minor_status = 0;
    return GSS_S_COMPLETE;
}


int
gss_oid_equal(const gss_OID a, const gss_OID b);

OM_uint32
heim_gss_import_name(OM_uint32 * /*minor_status*/,
		     const gss_buffer_t /*input_name_buffer*/,
		     const gss_OID /*input_name_type*/,
		     gss_name_t * /*output_name*/);



OM_uint32
gss_import_name(OM_uint32 *minor_status,
		gss_buffer_t input_name_buffer,
		gss_OID name_type,
		gss_name_t *name)
{
    LOG_ENTRY();

    /*
     * Rewrite gss_nt_krb5_principal
     */

    if (gss_oid_equal(name_type, (gss_OID)gss_nt_krb5_principal)) {
	struct comb_principal **p = (void *)input_name_buffer->value;
	input_name_buffer->value = &(*p)->heim;
    }

    return heim_gss_import_name(minor_status, input_name_buffer, name_type, name);
}
