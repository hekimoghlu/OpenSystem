/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 10, 2022.
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
#include "rquota.h"
#include <memory.h>
#include <sys/cdefs.h>
#ifndef __lint__
/*static char sccsid[] = "from: @(#)rquota.x 1.2 87/09/20 Copyr 1987 Sun Micro";*/
/*static char sccsid[] = "from: @(#)rquota.x	2.1 88/08/01 4.0 RPCSRC";*/
__RCSID("$NetBSD: rquota.x,v 1.6 2004/07/01 22:52:34 kleink Exp $");
#endif /* not __lint__ */

/* Default timeout can be changed using clnt_control() */
static struct timeval TIMEOUT = { 25, 0 };

void *
rquotaproc_null_1(void *argp, CLIENT *clnt)
{
	static char clnt_res;

	memset((char *)&clnt_res, 0, sizeof(clnt_res));
	if (clnt_call(clnt, RQUOTAPROC_NULL, (xdrproc_t)xdr_void, argp, (xdrproc_t)xdr_void, &clnt_res, TIMEOUT) != RPC_SUCCESS) {
		return NULL;
	}
	return (void *)&clnt_res;
}

getquota_rslt *
rquotaproc_getquota_1(getquota_args *argp, CLIENT *clnt)
{
	static getquota_rslt clnt_res;

	memset((char *)&clnt_res, 0, sizeof(clnt_res));
	if (clnt_call(clnt, RQUOTAPROC_GETQUOTA, (xdrproc_t)xdr_getquota_args, argp, (xdrproc_t)xdr_getquota_rslt, &clnt_res, TIMEOUT) != RPC_SUCCESS) {
		return NULL;
	}
	return &clnt_res;
}

getquota_rslt *
rquotaproc_getactivequota_1(getquota_args *argp, CLIENT *clnt)
{
	static getquota_rslt clnt_res;

	memset((char *)&clnt_res, 0, sizeof(clnt_res));
	if (clnt_call(clnt, RQUOTAPROC_GETACTIVEQUOTA, (xdrproc_t)xdr_getquota_args, argp, (xdrproc_t)xdr_getquota_rslt, &clnt_res, TIMEOUT) != RPC_SUCCESS) {
		return NULL;
	}
	return &clnt_res;
}

void *
rquotaproc_null_2(void *argp, CLIENT *clnt)
{
	static char clnt_res;

	memset((char *)&clnt_res, 0, sizeof(clnt_res));
	if (clnt_call(clnt, RQUOTAPROC_NULL, (xdrproc_t)xdr_void, argp, (xdrproc_t)xdr_void, &clnt_res, TIMEOUT) != RPC_SUCCESS) {
		return NULL;
	}
	return (void *)&clnt_res;
}

getquota_rslt *
rquotaproc_getquota_2(ext_getquota_args *argp, CLIENT *clnt)
{
	static getquota_rslt clnt_res;

	memset((char *)&clnt_res, 0, sizeof(clnt_res));
	if (clnt_call(clnt, RQUOTAPROC_GETQUOTA, (xdrproc_t)xdr_ext_getquota_args, argp, (xdrproc_t)xdr_getquota_rslt, &clnt_res, TIMEOUT) != RPC_SUCCESS) {
		return NULL;
	}
	return &clnt_res;
}

getquota_rslt *
rquotaproc_getactivequota_2(ext_getquota_args *argp, CLIENT *clnt)
{
	static getquota_rslt clnt_res;

	memset((char *)&clnt_res, 0, sizeof(clnt_res));
	if (clnt_call(clnt, RQUOTAPROC_GETACTIVEQUOTA, (xdrproc_t)xdr_ext_getquota_args, argp, (xdrproc_t)xdr_getquota_rslt, &clnt_res, TIMEOUT) != RPC_SUCCESS) {
		return NULL;
	}
	return &clnt_res;
}
