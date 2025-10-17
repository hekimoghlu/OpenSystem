/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 6, 2022.
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
#include "heimcred.h"
#include <gssapi_spi.h>

void
_gss_ntlm_iter_creds_f(OM_uint32 flags,
		       void *userctx ,
		       void (*cred_iter)(void *, gss_OID, gss_cred_id_t))
{
#ifdef HAVE_KCM
    krb5_error_code ret;
    krb5_context context = NULL;
    krb5_storage *request, *response;
    krb5_data response_data;
    
    ret = krb5_init_context(&context);
    if (ret)
	goto done;

    ret = krb5_kcm_storage_request(context, KCM_OP_GET_NTLM_USER_LIST, &request);
    if (ret)
	goto done;

    ret = krb5_kcm_call(context, request, &response, &response_data);
    krb5_storage_free(request);
    if (ret)
	goto done;

    while (1) {
	uint32_t morep;
	char *user = NULL, *domain = NULL;
	kcmuuid_t uuid;
	ntlm_cred dn;
	size_t sret;

	ret = krb5_ret_uint32(response, &morep);
	if (ret) goto out;

	if (!morep) goto out;

	ret = krb5_ret_stringz(response, &user);
	if (ret) goto out;
	ret = krb5_ret_stringz(response, &domain);
	if (ret) {
	    free(user);
	    goto out;
	}
	sret = krb5_storage_read(response, uuid, sizeof(uuid));
	if (sret != sizeof(uuid)) {
	    free(user);
	    free(domain);
	    goto out;
	}

	dn = calloc(1, sizeof(*dn));
	if (dn == NULL) {
	    free(user);
	    free(domain);
	    goto out;
	}
	dn->user = user;
	dn->domain = domain;
	dn->flags = NTLM_UUID;
	memcpy(dn->uuid, uuid, sizeof(dn->uuid));

	cred_iter(userctx, GSS_NTLM_MECHANISM, (gss_cred_id_t)dn);
    }
 out:
    krb5_storage_free(response);
    krb5_data_free(&response_data);
 done:
    if (context)
	krb5_free_context(context);
    (*cred_iter)(userctx, NULL, NULL);
#else /* !HAVE_KCM */

    CFDictionaryRef query = NULL;
    CFArrayRef query_result = NULL;

    const void *add_keys[] = {
    (void *)kHEIMObjectType,
	    kHEIMAttrType,
    };
    const void *add_values[] = {
    (void *)kHEIMObjectNTLM,
	    kHEIMTypeNTLM,
    };

    query = CFDictionaryCreate(NULL, add_keys, add_values, sizeof(add_keys) / sizeof(add_keys[0]), &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
    if (query == NULL)
	errx(1, "out of memory");

    query_result = HeimCredCopyQuery(query);
    CFRELEASE_NULL(query);

    CFIndex n, count = CFArrayGetCount(query_result);
    for (n = 0; n < count; n++) {
	char *user = NULL, *domain = NULL;

	HeimCredRef cred = (HeimCredRef)CFArrayGetValueAtIndex(query_result, n);
	CFStringRef userName = HeimCredCopyAttribute(cred, kHEIMAttrNTLMUsername);
	if (userName) {
	    user = rk_cfstring2cstring(userName);
	}
	CFStringRef domainName = HeimCredCopyAttribute(cred, kHEIMAttrNTLMDomain);
	if (domainName) {
	    domain = rk_cfstring2cstring(domainName);
	}

	CFUUIDBytes uuid_bytes;
	CFUUIDRef uuid_cfuuid = HeimCredGetUUID(cred);
	if (uuid_cfuuid) {
	    uuid_bytes = CFUUIDGetUUIDBytes(uuid_cfuuid);
	}

	ntlm_cred dn;

	dn = calloc(1, sizeof(*dn));
	if (dn == NULL) {
	    free(user);
	    free(domain);
	    CFRELEASE_NULL(userName);
	    CFRELEASE_NULL(domainName);
	    continue;
	}

	if (user == NULL || domain == NULL || uuid_cfuuid == NULL) {
	    free(dn);
	    free(user);
	    free(domain);
	    CFRELEASE_NULL(userName);
	    CFRELEASE_NULL(domainName);
	    continue;
	}

	dn->user = strdup(user);
	dn->domain = strdup(domain);
	dn->flags = NTLM_UUID;
	memcpy(dn->uuid, &uuid_bytes, sizeof(dn->uuid));

	cred_iter(userctx, GSS_NTLM_MECHANISM, (gss_cred_id_t)dn);

	free(user);
	free(domain);
	CFRELEASE_NULL(userName);
	CFRELEASE_NULL(domainName);
    }
    CFRELEASE_NULL(query_result);

    (*cred_iter)(userctx, NULL, NULL);
#endif /* HAVE_KCM */
}		 
