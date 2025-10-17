/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 23, 2024.
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
#include <stdio.h>
#include <strings.h>
#include <stdlib.h>
#include <gssapi/gssapi.h>
#include <gssapi/gssapi_krb5.h>
#include "lucid.h"
#include "gssd.h"

int
make_lucid_stream(gss_krb5_lucid_context_v1_t *lctx, size_t *len, void **data)
{
	lucid_context lucid;
	uint8_t *bufptr;
	uint32_t size;
	XDR xdrs;

	*len = 0;
	*data = NULL;
	switch (lctx->version) {
	case 1:
		break;
	default:
		return (FALSE);
	}
	lucid.vers = lctx->version;
	lucid.initiate = lctx->initiate;
	lucid.end_time = lctx->endtime;
	lucid.send_seq = lctx->send_seq;
	lucid.recv_seq = lctx->recv_seq;
	lucid.key_data.proto = lctx->protocol;
	switch (lctx->protocol) {
	case 0:
		lucid.key_data.lucid_protocol_u.data_1964.sign_alg = lctx->rfc1964_kd.sign_alg;
		lucid.key_data.lucid_protocol_u.data_1964.seal_alg = lctx->rfc1964_kd.seal_alg;
		lucid.ctx_key.etype = lctx->rfc1964_kd.ctx_key.type;
		lucid.ctx_key.key.key_len = lctx->rfc1964_kd.ctx_key.length;
		lucid.ctx_key.key.key_val = lctx->rfc1964_kd.ctx_key.data;
		break;
	case 1:
		lucid.key_data.lucid_protocol_u.data_4121.acceptor_subkey = lctx->cfx_kd.have_acceptor_subkey;
		if (lctx->cfx_kd.have_acceptor_subkey) {
			lucid.ctx_key.etype = lctx->cfx_kd.acceptor_subkey.type;
			lucid.ctx_key.key.key_len = lctx->cfx_kd.acceptor_subkey.length;
			lucid.ctx_key.key.key_val = lctx->cfx_kd.acceptor_subkey.data;
		} else {
			lucid.ctx_key.etype = lctx->cfx_kd.ctx_key.type;
			lucid.ctx_key.key.key_len = lctx->cfx_kd.ctx_key.length;
			lucid.ctx_key.key.key_val = lctx->cfx_kd.ctx_key.data;
		}
		break;
	default:
		return (FALSE);
	}
	size = xdr_sizeof((xdrproc_t)xdr_lucid_context, &lucid);
	if (size == 0)
		return (FALSE);
	bufptr = malloc(size);
	if (bufptr == NULL)
		return (FALSE);
	xdrmem_create(&xdrs, bufptr, size, XDR_ENCODE);
	if (xdr_lucid_context(&xdrs, &lucid)) {
		*len = size;
		*data = bufptr;
		return (TRUE);
	}
	return (FALSE);
}
