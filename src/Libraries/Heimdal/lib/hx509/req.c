/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 12, 2023.
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
#include "hx_locl.h"
#include <pkcs10_asn1.h>

struct hx509_request_data {
    hx509_name name;
    SubjectPublicKeyInfo key;
    ExtKeyUsage eku;
    GeneralNames san;
};

/*
 *
 */

int
hx509_request_init(hx509_context context, hx509_request *req)
{
    *req = calloc(1, sizeof(**req));
    if (*req == NULL)
	return ENOMEM;

    return 0;
}

void
hx509_request_free(hx509_request *req)
{
    if ((*req)->name)
	hx509_name_free(&(*req)->name);
    free_SubjectPublicKeyInfo(&(*req)->key);
    free_ExtKeyUsage(&(*req)->eku);
    free_GeneralNames(&(*req)->san);
    memset(*req, 0, sizeof(**req));
    free(*req);
    *req = NULL;
}

int
hx509_request_set_name(hx509_context context,
			hx509_request req,
			hx509_name name)
{
    if (req->name)
	hx509_name_free(&req->name);
    if (name) {
	int ret = hx509_name_copy(context, name, &req->name);
	if (ret)
	    return ret;
    }
    return 0;
}

int
hx509_request_get_name(hx509_context context,
			hx509_request req,
			hx509_name *name)
{
    if (req->name == NULL) {
	hx509_set_error_string(context, 0, EINVAL, "Request have no name");
	return EINVAL;
    }
    return hx509_name_copy(context, req->name, name);
}

int
hx509_request_set_SubjectPublicKeyInfo(hx509_context context,
					hx509_request req,
					const SubjectPublicKeyInfo *key)
{
    free_SubjectPublicKeyInfo(&req->key);
    return copy_SubjectPublicKeyInfo(key, &req->key);
}

int
hx509_request_get_SubjectPublicKeyInfo(hx509_context context,
					hx509_request req,
					SubjectPublicKeyInfo *key)
{
    return copy_SubjectPublicKeyInfo(&req->key, key);
}

int
_hx509_request_add_eku(hx509_context context,
		       hx509_request req,
		       const heim_oid *oid)
{
    void *val;
    int ret;

    val = realloc(req->eku.val, sizeof(req->eku.val[0]) * (req->eku.len + 1));
    if (val == NULL)
	return ENOMEM;
    req->eku.val = val;

    ret = der_copy_oid(oid, &req->eku.val[req->eku.len]);
    if (ret)
	return ret;

    req->eku.len += 1;

    return 0;
}

int
_hx509_request_add_dns_name(hx509_context context,
			    hx509_request req,
			    const char *hostname)
{
    GeneralName name;

    memset(&name, 0, sizeof(name));
    name.element = choice_GeneralName_dNSName;
    name.u.dNSName.data = rk_UNCONST(hostname);
    name.u.dNSName.length = strlen(hostname);

    return add_GeneralNames(&req->san, &name);
}

int
_hx509_request_add_email(hx509_context context,
			 hx509_request req,
			 const char *email)
{
    GeneralName name;

    memset(&name, 0, sizeof(name));
    name.element = choice_GeneralName_rfc822Name;
    name.u.dNSName.data = rk_UNCONST(email);
    name.u.dNSName.length = strlen(email);

    return add_GeneralNames(&req->san, &name);
}



int
_hx509_request_to_pkcs10(hx509_context context,
			 const hx509_request req,
			 const hx509_private_key signer,
			 heim_octet_string *request)
{
    CertificationRequest r;
    heim_octet_string data, os;
    int ret;
    size_t size = 0;

    if (req->name == NULL) {
	hx509_set_error_string(context, 0, EINVAL,
			       "PKCS10 needs to have a subject");
	return EINVAL;
    }

    memset(&r, 0, sizeof(r));
    memset(request, 0, sizeof(*request));

    r.certificationRequestInfo.version = pkcs10_v1;

    ret = copy_Name(&req->name->der_name,
		    &r.certificationRequestInfo.subject);
    if (ret)
	goto out;
    ret = copy_SubjectPublicKeyInfo(&req->key,
				    &r.certificationRequestInfo.subjectPKInfo);
    if (ret)
	goto out;
    r.certificationRequestInfo.attributes =
	calloc(1, sizeof(*r.certificationRequestInfo.attributes));
    if (r.certificationRequestInfo.attributes == NULL) {
	ret = ENOMEM;
	goto out;
    }

    ASN1_MALLOC_ENCODE(CertificationRequestInfo, data.data, data.length,
		       &r.certificationRequestInfo, &size, ret);
    if (ret)
	goto out;
    if (data.length != size)
	abort();

    ret = _hx509_create_signature(context,
				  signer,
				  _hx509_crypto_default_sig_alg,
				  &data,
				  &r.signatureAlgorithm,
				  &os);
    free(data.data);
    if (ret)
	goto out;
    r.signature.data = os.data;
    r.signature.length = os.length * 8;

    ASN1_MALLOC_ENCODE(CertificationRequest, data.data, data.length,
		       &r, &size, ret);
    if (ret)
	goto out;
    if (data.length != size)
	abort();

    *request = data;

out:
    free_CertificationRequest(&r);

    return ret;
}

int
_hx509_request_parse(hx509_context context,
		     const char *path,
		     hx509_request *req)
{
    CertificationRequest r;
    CertificationRequestInfo *rinfo;
    hx509_name subject;
    size_t len, size;
    void *p;
    int ret;

    if (strncmp(path, "PKCS10:", 7) != 0) {
	hx509_set_error_string(context, 0, HX509_UNSUPPORTED_OPERATION,
			       "unsupport type in %s", path);
	return HX509_UNSUPPORTED_OPERATION;
    }
    path += 7;

    /* XXX PEM request */

    ret = rk_undumpdata(path, &p, &len);
    if (ret) {
	hx509_set_error_string(context, 0, ret, "Failed to map file %s", path);
	return ret;
    }

    ret = decode_CertificationRequest(p, len, &r, &size);
    rk_xfree(p);
    if (ret) {
	hx509_set_error_string(context, 0, ret, "Failed to decode %s", path);
	return ret;
    }

    ret = hx509_request_init(context, req);
    if (ret) {
	free_CertificationRequest(&r);
	return ret;
    }

    rinfo = &r.certificationRequestInfo;

    ret = hx509_request_set_SubjectPublicKeyInfo(context, *req,
						  &rinfo->subjectPKInfo);
    if (ret) {
	free_CertificationRequest(&r);
	hx509_request_free(req);
	return ret;
    }

    ret = hx509_name_from_Name(&rinfo->subject, &subject);
    if (ret) {
	free_CertificationRequest(&r);
	hx509_request_free(req);
	return ret;
    }
    ret = hx509_request_set_name(context, *req, subject);
    hx509_name_free(&subject);
    free_CertificationRequest(&r);
    if (ret) {
	hx509_request_free(req);
	return ret;
    }

    return 0;
}


int
_hx509_request_print(hx509_context context, hx509_request req, FILE *f)
{
    int ret;

    if (req->name) {
	char *subject;
	ret = hx509_name_to_string(req->name, &subject);
	if (ret) {
	    hx509_set_error_string(context, 0, ret, "Failed to print name");
	    return ret;
	}
        fprintf(f, "name: %s\n", subject);
	free(subject);
    }

    return 0;
}

