/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 27, 2024.
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
#include <errno.h>
#include <sys/resource.h>
#include <mach/port.h>

extern int __iopolicysys(int, struct _iopol_param_t *);
extern void _pthread_clear_qos_tsd(mach_port_t);

int
getiopolicy_np(int iotype, int scope)
{
	int policy, error;
	struct _iopol_param_t iop_param;

	/* Do not sanity check iotype and scope, leave it to kernel. */

	iop_param.iop_scope = scope;
	iop_param.iop_iotype = iotype;
	error = __iopolicysys(IOPOL_CMD_GET, &iop_param);
	if (error != 0) {
		errno = error;
		policy = -1;
		goto exit;
	}

	policy = iop_param.iop_policy;

exit:
	return policy;
}

int
setiopolicy_np(int iotype, int scope, int policy)
{
	/* kernel validates the indiv values, no need to repeat it */
	struct _iopol_param_t iop_param;

	iop_param.iop_scope = scope;
	iop_param.iop_iotype = iotype;
	iop_param.iop_policy = policy;

	int rv = __iopolicysys(IOPOL_CMD_SET, &iop_param);
	if (rv == -2) {
		/* not an actual error but indication that __iopolicysys removed the thread QoS */
		_pthread_clear_qos_tsd(MACH_PORT_NULL);
		rv = 0;
	}

	return rv;
}
