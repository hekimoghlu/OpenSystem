/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 8, 2021.
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
#include <memory.h>
#include "mountd_rpc.h"

/* Default timeout can be changed using clnt_control() */
static struct timeval TIMEOUT = { 25, 0 };

void *
mountproc3_null_3(void *argp, CLIENT *clnt)
{
	static char clnt_res;

	memset((char *)&clnt_res, 0, sizeof(clnt_res));
	if (clnt_call(clnt, MOUNTPROC3_NULL, (xdrproc_t)xdr_void, argp, (xdrproc_t)xdr_void, &clnt_res, TIMEOUT) != RPC_SUCCESS) {
		return NULL;
	}
	return (void *)&clnt_res;
}

mountres3 *
mountproc3_mnt_3(dirpath *argp, CLIENT *clnt)
{
	static mountres3 clnt_res;

	memset((char *)&clnt_res, 0, sizeof(clnt_res));
	if (clnt_call(clnt, MOUNTPROC3_MNT, (xdrproc_t)xdr_dirpath, argp, (xdrproc_t)xdr_mountres3, &clnt_res, TIMEOUT) != RPC_SUCCESS) {
		return NULL;
	}
	return &clnt_res;
}

mountlist *
mountproc3_dump_3(void *argp, CLIENT *clnt)
{
	static mountlist clnt_res;

	memset((char *)&clnt_res, 0, sizeof(clnt_res));
	if (clnt_call(clnt, MOUNTPROC3_DUMP, (xdrproc_t)xdr_void, argp, (xdrproc_t)xdr_mountlist, &clnt_res, TIMEOUT) != RPC_SUCCESS) {
		return NULL;
	}
	return &clnt_res;
}

void *
mountproc3_umnt_3(dirpath *argp, CLIENT *clnt)
{
	static char clnt_res;

	memset((char *)&clnt_res, 0, sizeof(clnt_res));
	if (clnt_call(clnt, MOUNTPROC3_UMNT, (xdrproc_t)xdr_dirpath, argp, (xdrproc_t)xdr_void, &clnt_res, TIMEOUT) != RPC_SUCCESS) {
		return NULL;
	}
	return (void *)&clnt_res;
}

void *
mountproc3_umntall_3(void *argp, CLIENT *clnt)
{
	static char clnt_res;

	memset((char *)&clnt_res, 0, sizeof(clnt_res));
	if (clnt_call(clnt, MOUNTPROC3_UMNTALL, (xdrproc_t)xdr_void, argp, (xdrproc_t)xdr_void, &clnt_res, TIMEOUT) != RPC_SUCCESS) {
		return NULL;
	}
	return (void *)&clnt_res;
}

exports *
mountproc3_export_3(void *argp, CLIENT *clnt)
{
	static exports clnt_res;

	memset((char *)&clnt_res, 0, sizeof(clnt_res));
	if (clnt_call(clnt, MOUNTPROC3_EXPORT, (xdrproc_t)xdr_void, argp, (xdrproc_t)xdr_exports, &clnt_res, TIMEOUT) != RPC_SUCCESS) {
		return NULL;
	}
	return &clnt_res;
}
