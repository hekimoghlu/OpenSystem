/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 2, 2024.
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
 * catch_exception_raise_state will be implemented by user programs
 * This implementation is provided to resolve the reference in
 * exc_server().
 */

#include <mach/boolean.h>
#include <mach/message.h>
#include <mach/exception.h>
#include <mach/mig_errors.h>

#include "abort.h"
#include "exc_catcher.h"

__private_extern__ kern_return_t
internal_catch_exception_raise_state(
	mach_port_t exception_port,
	exception_type_t exception,
	exception_data_t code,
	mach_msg_type_number_t codeCnt,
	int *flavor,
	thread_state_t old_state,
	mach_msg_type_number_t old_stateCnt,
	thread_state_t new_state,
	mach_msg_type_number_t *new_stateCnt)
{
#if defined(__DYNAMIC__)
	static _libkernel_exc_raise_state_func_t exc_raise_state_func = (void*)-1;

	if (exc_raise_state_func == ((void*)-1) && _dlsym) {
		exc_raise_state_func = _dlsym(RTLD_DEFAULT, "catch_exception_raise_state");
	}
	if (exc_raise_state_func == 0) {
		/* The user hasn't defined catch_exception_raise in their binary */
		abort();
	}
	return (*exc_raise_state_func)(exception_port, exception, code, codeCnt, flavor, old_state, old_stateCnt, new_state, new_stateCnt);
#else
	extern kern_return_t catch_exception_raise_state(mach_port_t, exception_type_t, exception_data_t, mach_msg_type_number_t, int *, thread_state_t, mach_msg_type_number_t, thread_state_t, mach_msg_type_number_t *);
	return catch_exception_raise_state(exception_port, exception, code, codeCnt, flavor, old_state, old_stateCnt, new_state, new_stateCnt);
#endif
}
