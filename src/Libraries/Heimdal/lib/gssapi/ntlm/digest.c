/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 28, 2024.
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
#include <heim-ipc.h>
#include <digest_asn1.h>

/*
 *
 */

struct ntlmdgst {
    heim_ipc ipc;
    char *domain;
    OM_uint32 flags;
    struct ntlm_buf key;
    krb5_data sessionkey;
};

static OM_uint32 dstg_destroy(OM_uint32 *, void *);

/*
 *
 */

static OM_uint32
dstg_alloc(OM_uint32 *minor, void **ctx)
{
    krb5_error_code ret;
    struct ntlmdgst *c;

    c = calloc(1, sizeof(*c));
    if (c == NULL) {
	*minor = ENOMEM;
	return GSS_S_FAILURE;
    }

    ret = heim_ipc_init_context("ANY:org.h5l.ntlm-service", &c->ipc);
    if (ret) {
	free(c);
	*minor = ENOMEM;
	return GSS_S_FAILURE;
    }

    *ctx = c;

    return GSS_S_COMPLETE;
}

static int
dstg_probe(OM_uint32 *minor_status, void *ctx, const char *realm, unsigned int *flags)
{
    struct ntlmdgst *c = ctx;
    heim_idata dreq, drep;
    NTLMInitReply ir;
    size_t size = 0;
    NTLMInit ni;
    int ret;

    memset(&ni, 0, sizeof(ni));
    
    ni.flags = 0;

    ASN1_MALLOC_ENCODE(NTLMInit, dreq.data, dreq.length, &ni, &size, ret);
    if (ret) {
	*minor_status = ret;
	return GSS_S_FAILURE;
    }
    if (size != dreq.length)
	abort();
    
    ret = heim_ipc_call(c->ipc, &dreq, &drep, NULL);
    free(dreq.data);
    if (ret) {
	*minor_status = ret;
	return GSS_S_FAILURE;
    }
    
    ret = decode_NTLMInitReply(drep.data, drep.length, &ir, &size);
    free(drep.data);
    if (ret) {
	*minor_status = ret;
	return GSS_S_FAILURE;
    }

    if (ir.ntlmNegFlags & NTLM_NEG_SIGN)
	*flags |= NSI_NO_SIGNING;

    free_NTLMInitReply(&ir);
    
    return 0;
}

/*
 *
 */

static OM_uint32
dstg_destroy(OM_uint32 *minor, void *ctx)
{
    struct ntlmdgst *c = ctx;
    krb5_data_free(&c->sessionkey);
    if (c->ipc)
	heim_ipc_free_context(c->ipc);
    memset(c, 0, sizeof(*c));
    free(c);

    return GSS_S_COMPLETE;
}

/*
 *
 */

static OM_uint32
dstg_ti(OM_uint32 *minor_status,
	ntlm_ctx ntlmctx,
	void *ctx,
	const char *hostname,
	const char *domain,
	uint32_t *negNtlmFlags)
{
    struct ntlmdgst *c = ctx;
    OM_uint32 maj_stat = GSS_S_FAILURE;
    heim_idata dreq, drep;
    NTLMInitReply ir;
    size_t size = 0;
    NTLMInit ni;
    int ret;

    memset(&ni, 0, sizeof(ni));
    memset(&ir, 0, sizeof(ir));

    ni.flags = 0;
    if (hostname)
	ni.hostname = (char **)&hostname;
    if (domain)
	ni.domain = (char **)&domain;

    ASN1_MALLOC_ENCODE(NTLMInit, dreq.data, dreq.length, &ni, &size, ret);
    if (ret) {
	*minor_status = ret;
	return GSS_S_FAILURE;
    }
    if (size != dreq.length)
	abort();

    ret = heim_ipc_call(c->ipc, &dreq, &drep, NULL);
    free(dreq.data);
    if (ret) {
	*minor_status = ret;
	return GSS_S_FAILURE;
    }

    ret = decode_NTLMInitReply(drep.data, drep.length, &ir, &size);
    free(drep.data);
    if (ret) {
	*minor_status = ret;
	return GSS_S_FAILURE;
    }

    {
	struct ntlm_buf buf;

	buf.data = ir.targetinfo.data;
	buf.length = ir.targetinfo.length;

	ret = heim_ntlm_decode_targetinfo(&buf, 1, &ntlmctx->ti);
	if (ret) {
	    free_NTLMInitReply(&ir);
	    *minor_status = ret;
	    return GSS_S_FAILURE;
	}
    }
    *negNtlmFlags = ir.ntlmNegFlags;

    maj_stat = GSS_S_COMPLETE;
    
    free_NTLMInitReply(&ir);

    return maj_stat;
}

/*
 *
 */

static OM_uint32
dstg_type3(OM_uint32 *minor_status,
	   ntlm_ctx ntlmctx,
	   void *ctx,
	   const struct ntlm_type3 *type3,
	   ntlm_cred acceptor_cred,
	   uint32_t *flags,
	   uint32_t *avflags,
	   struct ntlm_buf *sessionkey,
	   ntlm_name *name, struct ntlm_buf *uuid,
	   struct ntlm_buf *pac)
{
    struct ntlmdgst *c = ctx;
    krb5_error_code ret;
    NTLMRequest2 req;
    NTLMReply rep;
    heim_idata dreq, drep;
    size_t size = 0;
    
    *avflags = *flags = 0;

    sessionkey->data = NULL;
    sessionkey->length = 0;
    *name = NULL;
    uuid->data = NULL;
    uuid->length = 0;
    pac->data = NULL;
    pac->length = 0;

    memset(&req, 0, sizeof(req));
    memset(&rep, 0, sizeof(rep));

    req.loginUserName = type3->username;
    req.loginDomainName = type3->targetname;
    req.workstation = type3->ws;
    req.ntlmFlags = type3->flags;
    req.lmchallenge.data = ntlmctx->challenge;
    req.lmchallenge.length = sizeof(ntlmctx->challenge);
    req.ntChallengeResponse.data = type3->ntlm.data;
    req.ntChallengeResponse.length = type3->ntlm.length;
    req.lmChallengeResponse.data = type3->lm.data;
    req.lmChallengeResponse.length = type3->lm.length;
    req.encryptedSessionKey.data = type3->sessionkey.data;
    req.encryptedSessionKey.length = type3->sessionkey.length;
    req.t2targetname = ntlmctx->ti.domainname;
    if (acceptor_cred) {
	req.acceptorUser = acceptor_cred->user;
	req.acceptorDomain = acceptor_cred->domain;
    } else {
	req.acceptorUser = "";
	req.acceptorDomain = "";
    }

    /* take care of type3->targetname ? */

    ASN1_MALLOC_ENCODE(NTLMRequest2, dreq.data, dreq.length, &req, &size, ret);
    if (ret) {
	*minor_status = ret;
	return GSS_S_FAILURE;
    }
    if (size != dreq.length)
	abort();
	
    ret = heim_ipc_call(c->ipc, &dreq, &drep, NULL);
    free(dreq.data);
    if (ret) {
	*minor_status = ret;
	return gss_mg_set_error_string(GSS_NTLM_MECHANISM, GSS_S_FAILURE, ret,
				       "ipc to digest-service failed");
    }	

    ret = decode_NTLMReply(drep.data, drep.length, &rep, &size);
    free(drep.data);
    if (ret) {
	*minor_status = ret;
	return gss_mg_set_error_string(GSS_NTLM_MECHANISM, GSS_S_FAILURE, ret,
				       "message from digest-service malformed");
    }	
    
    if (rep.success != TRUE) {
	ret = HNTLM_ERR_AUTH;
	gss_mg_set_error_string(GSS_NTLM_MECHANISM,
				GSS_S_FAILURE, ret,
				"ntlm: authentication failed");
	goto out;
    }

    *flags = rep.ntlmFlags;
    *avflags = rep.avflags;

    if (rep.avflags & NTLM_TI_AV_FLAG_GUEST)
	*flags |= NTLM_NEG_ANONYMOUS;

    /* handle session key */
    if (rep.sessionkey) {
	sessionkey->data = malloc(rep.sessionkey->length);
	memcpy(sessionkey->data, rep.sessionkey->data,
	       rep.sessionkey->length);
	sessionkey->length = rep.sessionkey->length;
    }
    
    *name = calloc(1, sizeof(**name));
    if (*name == NULL)
	goto out;
    (*name)->user = strdup(rep.user);
    (*name)->domain = strdup(rep.domain);
    if ((*name)->user == NULL || (*name)->domain == NULL)
	goto out;

    if (rep.uuid) {
	uuid->data = malloc(rep.uuid->length);
	memcpy(uuid->data, rep.uuid->data, rep.uuid->length);
	uuid->length = rep.uuid->length;
    }

    free_NTLMReply(&rep);

    return 0;

 out:
    free_NTLMReply(&rep);
    *minor_status = ret;
    return GSS_S_FAILURE;
}

/*
 *
 */

static void
dstg_free_buffer(struct ntlm_buf *sessionkey)
{
    if (sessionkey->data)
	free(sessionkey->data);
    sessionkey->data = NULL;
    sessionkey->length = 0;
}

/*
 *
 */

struct ntlm_server_interface ntlmsspi_dstg_digest = {
    "digest",
    dstg_alloc,
    dstg_destroy,
    dstg_probe,
    dstg_type3,
    dstg_free_buffer,
    dstg_ti
};
