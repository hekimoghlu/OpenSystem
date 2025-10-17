/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 20, 2021.
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
#include <sys/sysctl.h>

__private_extern__ void
skmem_sysctl_init(void)
{
	// TCP values
	skmem_sysctl *__single sysctls = skmem_get_sysctls_obj(NULL);
	if (sysctls) {
		sysctls->version = SKMEM_SYSCTL_VERSION;
#define X(type, field, default_value) \
	        extern struct sysctl_oid sysctl__net_inet_tcp_##field;                          \
	        sysctls->tcp.field = *(type*)sysctl__net_inet_tcp_##field.oid_arg1;
		SKMEM_SYSCTL_TCP_LIST
#undef  X
	}
}

__private_extern__ int
skmem_sysctl_handle_int(__unused struct sysctl_oid *oidp, void *arg1,
    int arg2, struct sysctl_req *req)
{
	int changed = 0;
	int result = sysctl_io_number(req, *(int*)arg1, sizeof(int), arg1,
	    &changed);
	if (changed) {
		SYSCTL_SKMEM_UPDATE_AT_OFFSET(arg2, *(int*)arg1);
	}
	return result;
}
