/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 21, 2023.
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
#include <sys/param.h>
#include <sys/proc.h>
#include <sys/kauth.h>
#include <security/mac_framework.h>
#include <security/mac_internal.h>

int
mac_skywalk_flow_check_connect(__unused proc_t proc, void *flow, const struct sockaddr *addr, int type, int protocol)
{
	int error;

	assert(proc == current_proc());
	MAC_CHECK(skywalk_flow_check_connect, kauth_cred_get(), flow, addr, type, protocol);
	return error;
}

int
mac_skywalk_flow_check_listen(__unused proc_t proc, void *flow, const struct sockaddr *addr, int type, int protocol)
{
	int error;

	assert(proc == current_proc());
	MAC_CHECK(skywalk_flow_check_listen, kauth_cred_get(), flow, addr, type, protocol);
	return error;
}
