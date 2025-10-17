/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 2, 2025.
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
/*
 * Copyright (c) 1992, 1993, 1994
 *    The Regents of the University of California.  All rights reserved.
 *
 * This code is derived from software contributed to Berkeley by
 * Rick Macklem at The University of Guelph.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *    This product includes software developed by the University of
 *    California, Berkeley and its contributors.
 * 4. Neither the name of the University nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */

#include "sm_inter.h"

#import <XCTest/XCTest.h>

sm_stat_res *
STAT_doStatRPC(CLIENT *clnt, char *name)
{
	struct sm_name smname = {.mon_name = name};
	struct sm_stat_res *result = NULL;

	result = sm_stat_1(&smname, clnt);
	if (result == NULL) {
		XCTFail("sm_stat_1 returned null");
	}
	return result;
}

sm_stat_res *
STAT_doMonRPC(CLIENT *clnt, char *mon_name, char *my_name, int my_prog, int my_vers, int my_proc, char *priv)
{
	struct mon_id mon_id = {.mon_name = mon_name,
		                .my_id = {.my_name = my_name, .my_proc = my_proc, .my_prog = my_prog, .my_vers = my_vers}};
	struct mon mon = {.mon_id = mon_id, .priv = {0}};
	strncpy(mon.priv, priv, sizeof(mon.priv));
	struct sm_stat_res *result = NULL;

	result = sm_mon_1(&mon, clnt);
	if (result == NULL) {
		XCTFail("sm_mon_1 returned null");
	}
	return result;
}

sm_stat *
STAT_doUnmonRPC(CLIENT *clnt, char *mon_name, char *my_name, int my_prog, int my_vers, int my_proc)
{
	struct mon_id mon_id = {.mon_name = mon_name,
		                .my_id = {.my_name = my_name, .my_proc = my_proc, .my_prog = my_prog, .my_vers = my_vers}};

	struct sm_stat *result = NULL;

	result = sm_unmon_1(&mon_id, clnt);
	if (result == NULL) {
		XCTFail("sm_unmon_1 returned null");
	}
	return result;
}


sm_stat *
STAT_doUnmonAllRPC(CLIENT *clnt, char *mon_name, char *my_name, int my_prog, int my_vers, int my_proc)
{
	struct my_id my_id = {.my_name = my_name, .my_proc = my_proc, .my_prog = my_prog, .my_vers = my_vers};
	struct sm_stat *result = NULL;

	result = sm_unmon_all_1(&my_id, clnt);
	if (result == NULL) {
		XCTFail("sm_unmon_all_1 returned null");
	}
	return result;
}

sm_stat *
STAT_doNotifyRPC(CLIENT *clnt, char *mon_name, int state)
{
	struct stat_chge stat_chge = {.mon_name = mon_name, .state = state};
	void *result = NULL;

	result = sm_notify_1(&stat_chge, clnt);
	if (result == NULL) {
		XCTFail("sm_notify_1 returned null");
	}
	return result;
}
