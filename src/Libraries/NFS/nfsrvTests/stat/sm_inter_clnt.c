/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 26, 2024.
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
#include "sm_inter.h"
#include <sys/cdefs.h>
__RCSID("$FreeBSD: src/include/rpcsvc/sm_inter.x,v 1.11 2003/05/04 02:51:42 obrien Exp $");

/* Default timeout can be changed using clnt_control() */
static struct timeval TIMEOUT = { 25, 0 };

struct sm_stat_res *
sm_stat_1(struct sm_name *argp, CLIENT *clnt)
{
	static struct sm_stat_res clnt_res;

	memset((char *)&clnt_res, 0, sizeof(clnt_res));
	if (clnt_call(clnt, SM_STAT, (xdrproc_t)xdr_sm_name, argp, (xdrproc_t)xdr_sm_stat_res, &clnt_res, TIMEOUT) != RPC_SUCCESS) {
		return NULL;
	}
	return &clnt_res;
}

struct sm_stat_res *
sm_mon_1(struct mon *argp, CLIENT *clnt)
{
	static struct sm_stat_res clnt_res;

	memset((char *)&clnt_res, 0, sizeof(clnt_res));
	if (clnt_call(clnt, SM_MON, (xdrproc_t)xdr_mon, argp, (xdrproc_t)xdr_sm_stat_res, &clnt_res, TIMEOUT) != RPC_SUCCESS) {
		return NULL;
	}
	return &clnt_res;
}

struct sm_stat *
sm_unmon_1(struct mon_id *argp, CLIENT *clnt)
{
	static struct sm_stat clnt_res;

	memset((char *)&clnt_res, 0, sizeof(clnt_res));
	if (clnt_call(clnt, SM_UNMON, (xdrproc_t)xdr_mon_id, argp, (xdrproc_t)xdr_sm_stat, &clnt_res, TIMEOUT) != RPC_SUCCESS) {
		return NULL;
	}
	return &clnt_res;
}

struct sm_stat *
sm_unmon_all_1(struct my_id *argp, CLIENT *clnt)
{
	static struct sm_stat clnt_res;

	memset((char *)&clnt_res, 0, sizeof(clnt_res));
	if (clnt_call(clnt, SM_UNMON_ALL, (xdrproc_t)xdr_my_id, argp, (xdrproc_t)xdr_sm_stat, &clnt_res, TIMEOUT) != RPC_SUCCESS) {
		return NULL;
	}
	return &clnt_res;
}

void *
sm_simu_crash_1(void *argp, CLIENT *clnt)
{
	static char clnt_res;

	memset((char *)&clnt_res, 0, sizeof(clnt_res));
	if (clnt_call(clnt, SM_SIMU_CRASH, (xdrproc_t)xdr_void, argp, (xdrproc_t)xdr_void, &clnt_res, TIMEOUT) != RPC_SUCCESS) {
		return NULL;
	}
	return (void *)&clnt_res;
}

void *
sm_notify_1(struct stat_chge *argp, CLIENT *clnt)
{
	static char clnt_res;

	memset((char *)&clnt_res, 0, sizeof(clnt_res));
	if (clnt_call(clnt, SM_NOTIFY, (xdrproc_t)xdr_stat_chge, argp, (xdrproc_t)xdr_void, &clnt_res, TIMEOUT) != RPC_SUCCESS) {
		return NULL;
	}
	return (void *)&clnt_res;
}
