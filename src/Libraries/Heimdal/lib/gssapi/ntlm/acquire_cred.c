/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 17, 2022.
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
#include <gssapi_spi.h>
#include "heimcred.h"
#include "heimbase.h"

OM_uint32
_gss_ntlm_have_cred(OM_uint32 *minor,
		    const ntlm_name target_name,
		    ntlm_cred *rcred)
{
    krb5_context context;
    krb5_error_code ret;
#ifdef HAVE_KCM
	krb5_storage *request, *response;
	krb5_data response_data;
	ssize_t sret;
#else /* !HAVE_KCM */
	CFMutableDictionaryRef query = NULL;
	CFArrayRef query_result = NULL;
	CFIndex query_count = 0;
	CFStringRef user_cfstr = NULL;
	CFStringRef domain_cfstr = NULL;
	CFUUIDRef uuid_cfuuid = NULL;
	CFUUIDBytes uuid_bytes;
	HeimCredRef result = NULL;
#endif /* HAVE_KCM */
	OM_uint32 major = GSS_S_FAILURE;
	ntlm_name cred;
	kcmuuid_t uuid;

    ret = krb5_init_context(&context);
    if (ret) {
	*minor = ret;
	return GSS_S_FAILURE;
    }
#ifdef HAVE_KCM
    ret = krb5_kcm_storage_request(context, KCM_OP_HAVE_NTLM_CRED, &request);
    if (ret)
	goto out;

    ret = krb5_store_stringz(request, target_name->user);
    if (ret)
	goto out;

    ret = krb5_store_stringz(request, target_name->domain);
    if (ret)
	goto out;

    ret = krb5_kcm_call(context, request, &response, &response_data);
    krb5_storage_free(request);
    if (ret)
	goto out;

    sret = krb5_storage_read(response, uuid, sizeof(uuid));

    krb5_storage_free(response);
    krb5_data_free(&response_data);

    if (sret != sizeof(uuid)) {
	krb5_clear_error_message(context);
	ret = KRB5_CC_IO;
	goto out;
    }
    
    major = _gss_ntlm_duplicate_name(minor, (gss_name_t)target_name,
				     (gss_name_t *)&cred);
    if (major)
	goto out;
#else /* !HAVE_KCM */
	user_cfstr = CFStringCreateWithCString(kCFAllocatorDefault, target_name->user,kCFStringEncodingUTF8);
	if (user_cfstr == NULL)
		goto out;
	domain_cfstr = CFStringCreateWithCString(kCFAllocatorDefault, target_name->domain,kCFStringEncodingUTF8);
	if (domain_cfstr == NULL)
		goto out;

	const void *add_keys[] = {
	    (void *)kHEIMObjectType,
	    kHEIMAttrType,
	    kHEIMAttrParentCredential,
	};
	const void *add_values[] = {
	    (void *)kHEIMObjectNTLM,
	    kHEIMTypeNTLM,
	    kCFNull,  // do not return labels as creds
	};

	CFDictionaryRef baseQuery = CFDictionaryCreate(NULL, add_keys, add_values, sizeof(add_keys) / sizeof(add_keys[0]), &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
	heim_assert(baseQuery != NULL, "Failed to create dictionary");
	
    query = CFDictionaryCreateMutableCopy(NULL, 0, baseQuery);
    CFRELEASE_NULL(baseQuery);
    
    if (CFStringGetLength(user_cfstr) > 0) {
	CFDictionaryAddValue(query, kHEIMAttrNTLMUsername, user_cfstr);
    }
    
    if (CFStringGetLength(domain_cfstr) > 0) {
	CFDictionaryAddValue(query, kHEIMAttrNTLMDomain, domain_cfstr);
    }
    
	query_result = HeimCredCopyQuery(query);

	if (query_result == NULL)
		goto out;

	query_count = CFArrayGetCount(query_result);
	if (query_count == 0) {
		goto out;
	}
	result = (HeimCredRef) CFArrayGetValueAtIndex(query_result, 0);
	if (result == NULL) {
	    goto out;
	}
	uuid_cfuuid = HeimCredGetUUID(result);
	if (uuid_cfuuid == NULL)
		goto out;
	uuid_bytes = CFUUIDGetUUIDBytes(uuid_cfuuid);
	memcpy(&uuid, &uuid_bytes, sizeof (uuid));
	uuid_string_t uuid_cstr;
	uuid_unparse(uuid, uuid_cstr);
	_gss_mg_log(1, "_gss_ntlm_have_cred  UUID(%s)", uuid_cstr);
    
    // If the query was for any cred, then make a new cred matching the GSSCred response
    if (CFStringGetLength(user_cfstr) == 0 || CFStringGetLength(domain_cfstr) == 0) {
	
	cred = calloc(1, sizeof(*cred));
	if (cred == NULL) {
	    major = GSS_S_FAILURE;
	    goto out;
	}
	
	CFStringRef userName = HeimCredCopyAttribute(result, kHEIMAttrNTLMUsername);
	if (userName) {
	    cred->user = rk_cfstring2cstring(userName);
	    CFRELEASE_NULL(userName);
	}
	
	CFStringRef domainName = HeimCredCopyAttribute(result, kHEIMAttrNTLMDomain);
	if (domainName) {
	    cred->domain = rk_cfstring2cstring(domainName);
	    CFRELEASE_NULL(domainName);
	}
	
	if (cred->user == NULL || cred->domain == NULL) {
	    gss_name_t tempn =  (gss_name_t)cred;
	    _gss_ntlm_release_name(minor, &tempn);
	    ret = ENOMEM;
	    goto out;
	}
    } else {
	major = _gss_ntlm_duplicate_name(minor, (gss_name_t)target_name,
					 (gss_name_t *)&cred);
	if (major)
	    goto out;
    }
#endif /* HAVE_KCM */

    cred->flags |= NTLM_UUID;
    memcpy(cred->uuid, uuid, sizeof(cred->uuid));

    *rcred = (ntlm_cred)cred;
    
    major = GSS_S_COMPLETE;
    
 out:
    krb5_free_context(context);
    if (ret) {
	*minor = ret;
	major = GSS_S_FAILURE;
    }
#ifndef HAVE_KCM
	if (user_cfstr)
		CFRelease(user_cfstr);
	if (domain_cfstr)
		CFRelease(domain_cfstr);
	if (query)
		CFRelease(query);
	if (query_result)
	    CFRelease(query_result);
#endif /* HAVE_KCM */
    return major;
}

OM_uint32
_gss_ntlm_acquire_cred(OM_uint32 * min_stat,
		       const gss_name_t desired_name,
		       OM_uint32 time_req,
		       const gss_OID_set desired_mechs,
		       gss_cred_usage_t cred_usage,
		       gss_cred_id_t * output_cred_handle,
		       gss_OID_set * actual_mechs,
		       OM_uint32 * time_rec)
{
    ntlm_name name = (ntlm_name) desired_name;
    OM_uint32 maj_stat, junk;
    ntlm_ctx ctx;

    *min_stat = 0;
    *output_cred_handle = GSS_C_NO_CREDENTIAL;
    if (actual_mechs)
	*actual_mechs = GSS_C_NO_OID_SET;
    if (time_rec)
	*time_rec = GSS_C_INDEFINITE;

    if (desired_name == NULL)
	return GSS_S_NO_CRED;

    if (cred_usage == GSS_C_BOTH || cred_usage == GSS_C_ACCEPT) {
	gss_ctx_id_t gctx;

	maj_stat = _gss_ntlm_allocate_ctx(min_stat, name->domain, &ctx);
	if (maj_stat != GSS_S_COMPLETE)
	    return maj_stat;

	gctx = (gss_ctx_id_t)ctx;
	_gss_ntlm_delete_sec_context(&junk, &gctx, NULL);
    }	
    if (cred_usage == GSS_C_BOTH || cred_usage == GSS_C_INITIATE) {
	ntlm_cred cred;

	/* if we have a anon name, lets dup it directly */
	if ((name->flags & NTLM_ANON_NAME) != 0) {
	    maj_stat = _gss_ntlm_duplicate_name(min_stat,
						(gss_name_t)name,
						(gss_name_t *)&cred);
	    if (maj_stat)
		return maj_stat;
	} else {
	    maj_stat = _gss_ntlm_have_cred(min_stat, name, &cred);
	    if (maj_stat)
		return maj_stat;
	}
	*output_cred_handle = (gss_cred_id_t)cred;
    }

    return (GSS_S_COMPLETE);
}

OM_uint32
_gss_ntlm_acquire_cred_ext(OM_uint32 * minor_status,
			   const gss_name_t desired_name,
			   gss_const_OID credential_type,
			   const void *credential_data,
			   OM_uint32 time_req,
			   gss_const_OID desired_mech,
			   gss_cred_usage_t cred_usage,
			   gss_cred_id_t * output_cred_handle)
{
    ntlm_name name = (ntlm_name) desired_name;
    OM_uint32 major = GSS_S_FAILURE;
    krb5_context context;
    krb5_error_code ret;
    struct ntlm_buf buf;
    krb5_data data;
    ntlm_cred dn;
    kcmuuid_t uuid;
#ifdef HAVE_KCM
    krb5_storage *request, *response;
    krb5_data response_data;
    ssize_t sret;
#else /* !HAVE_KCM */
	HeimCredRef cred = NULL;
	CFErrorRef cferr = NULL;
	CFStringRef user_cfstr = NULL;
	CFStringRef domain_cfstr = NULL;
	CFDataRef ntlmhash_cfdata = NULL;
	CFUUIDRef uuid_cfuuid = NULL;
	CFUUIDBytes uuid_bytes;
	CFDictionaryRef attrs = NULL;
#endif /* HAVE_KCM */
    if (credential_data == NULL)
	return GSS_S_FAILURE;

    if (!gss_oid_equal(credential_type, GSS_C_CRED_PASSWORD))
	return GSS_S_FAILURE;

    if (name == NULL)
	return GSS_S_FAILURE;
	
    ret = krb5_init_context(&context);
    if (ret)
	return GSS_S_FAILURE;

    {
	gss_buffer_t buffer;
	char *password;
	
	buffer = (gss_buffer_t)credential_data;
	password = malloc(buffer->length + 1);
	if (password == NULL) {
	    ret = ENOMEM;
	    goto out;
	}
	memcpy(password, buffer->value, buffer->length);
	password[buffer->length] = '\0';
	
	heim_ntlm_nt_key(password, &buf);
	memset(password, 0, strlen(password));
	free(password);
    }

    data.data = buf.data;
    data.length = buf.length;
#ifdef HAVE_KCM
    krb5_kcm_storage_request(context, KCM_OP_ADD_NTLM_CRED, &request);
	
    krb5_store_stringz(request, name->user);
    krb5_store_stringz(request, name->domain);
    krb5_store_data(request, data);
    
    ret = krb5_kcm_call(context, request, &response, &response_data);
    krb5_storage_free(request);
    if (ret)
	goto out;
    
    sret = krb5_storage_read(response, &uuid, sizeof(uuid));

    krb5_storage_free(response);
    krb5_data_free(&response_data);

    if (sret != sizeof(uuid)) {
	major = GSS_S_FAILURE;
	ret = KRB5_CC_IO;
	goto out;
    }
#else /* !HAVE_KCM */
	/* store in NTLM credential  cache */
	 /* username domain nthash <UUID> */
	user_cfstr = CFStringCreateWithCString(kCFAllocatorDefault, name->user,kCFStringEncodingUTF8);
	if (user_cfstr == NULL)
		goto out;
	domain_cfstr = CFStringCreateWithCString(kCFAllocatorDefault, name->domain,kCFStringEncodingUTF8);
	if (domain_cfstr == NULL)
		goto out;
	ntlmhash_cfdata = CFDataCreate(kCFAllocatorDefault, data.data, data.length);
	if (ntlmhash_cfdata == NULL)
		goto out;

	const void *add_keys[] = {
	(void *)kHEIMObjectType,
			kHEIMAttrType,
			kHEIMAttrNTLMUsername,
			kHEIMAttrNTLMDomain,
			kHEIMAttrData
	};
	const void *add_values[] = {
	(void *)kHEIMObjectNTLM,
			kHEIMTypeNTLM,
			user_cfstr,
			domain_cfstr,
			ntlmhash_cfdata
	};

	attrs = CFDictionaryCreate(NULL, add_keys, add_values, sizeof(add_keys) / sizeof(add_keys[0]), &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
	heim_assert(attrs != NULL, "Failed to create dictionary");

	// TODO check for duplicate in GSSCred? HeimCredCopyQuery <OR>  handled in _gss_ntlm_have_cred
	cred = HeimCredCreate(attrs, &cferr);
	if (cred == NULL)
		goto out;
	uuid_cfuuid = HeimCredGetUUID(cred);
	if (uuid_cfuuid == NULL)
		goto out;
	uuid_bytes = CFUUIDGetUUIDBytes(uuid_cfuuid);
	memcpy(&uuid, &uuid_bytes, sizeof (uuid));
	uuid_string_t uuid_cstr;
	uuid_unparse(uuid, uuid_cstr);
	_gss_mg_log(1, "_gss_ntlm_acquire_cred_ext name(%s) domain(%s) UUID(%s)", name->user, name->domain, uuid_cstr);
#endif /* HAVE_KCM */

    heim_ntlm_free_buf(&buf);

    dn = calloc(1, sizeof(*dn));
    if (dn == NULL) {
	major = GSS_S_FAILURE;
	goto out;
    }

    dn->user = strdup(name->user);
    dn->domain = strdup(name->domain);
    dn->flags = NTLM_UUID;
    memcpy(dn->uuid, uuid, sizeof(dn->uuid));

    *output_cred_handle = (gss_cred_id_t)dn;
    
    major = GSS_S_COMPLETE;
 out:

    krb5_free_context(context);
    if (ret)
	major = GSS_S_FAILURE;

#ifndef HAVE_KCM /* GSSCred */
	if (user_cfstr)
		CFRelease(user_cfstr);
	if (domain_cfstr)
		CFRelease(domain_cfstr);
	if (ntlmhash_cfdata)
		CFRelease(ntlmhash_cfdata);
	if (attrs)
		CFRelease(attrs);
	if (cred)
		CFRelease(cred);
#endif
    return major;
}
