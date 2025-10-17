/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 16, 2024.
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
#undef _thread_act_user_
#include <mach/thread_act_internal.h>

extern void _pthread_clear_qos_tsd(mach_port_t port);

kern_return_t
thread_policy(thread_act_t thr_act, policy_t policy, policy_base_t base, mach_msg_type_number_t baseCnt, boolean_t set_limit)
{
	kern_return_t kr;

	kr = _kernelrpc_thread_policy(thr_act, policy, base, baseCnt, set_limit);
	if (kr == KERN_SUCCESS) {
		_pthread_clear_qos_tsd(thr_act);
	} else if (kr == KERN_POLICY_STATIC) {
		kr = KERN_SUCCESS;
	}

	return kr;
}

kern_return_t
thread_policy_set(thread_act_t thread, thread_policy_flavor_t flavor, thread_policy_t policy_info, mach_msg_type_number_t policy_infoCnt)
{
	kern_return_t kr;

	kr = _kernelrpc_thread_policy_set(thread, flavor, policy_info, policy_infoCnt);
	if (kr == KERN_SUCCESS) {
		_pthread_clear_qos_tsd(thread);
	} else if (kr == KERN_POLICY_STATIC) {
		kr = KERN_SUCCESS;
	}

	return kr;
}

kern_return_t
thread_set_policy(thread_act_t thr_act, processor_set_t pset, policy_t policy, policy_base_t base, mach_msg_type_number_t baseCnt, policy_limit_t limit, mach_msg_type_number_t limitCnt)
{
	kern_return_t kr;

	kr = _kernelrpc_thread_set_policy(thr_act, pset, policy, base, baseCnt, limit, limitCnt);
	if (kr == KERN_SUCCESS) {
		_pthread_clear_qos_tsd(thr_act);
	} else if (kr == KERN_POLICY_STATIC) {
		kr = KERN_SUCCESS;
	}

	return kr;
}
