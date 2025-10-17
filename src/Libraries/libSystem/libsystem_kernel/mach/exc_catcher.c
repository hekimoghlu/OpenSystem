/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 15, 2022.
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
 * catch_exception_raise will be implemented by user programs
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
internal_catch_exception_raise(
	mach_port_t exception_port,
	mach_port_t thread,
	mach_port_t task,
	exception_type_t exception,
	exception_data_t code,
	mach_msg_type_number_t codeCnt)
{
#if defined(__DYNAMIC__)
	static _libkernel_exc_raise_func_t exc_raise_func = (void*)-1;

	if (exc_raise_func == ((void*)-1) && _dlsym) {
		exc_raise_func = _dlsym(RTLD_DEFAULT, "catch_exception_raise");
	}
	if (exc_raise_func == 0) {
		/* The user hasn't defined catch_exception_raise in their binary */
		abort();
	}
	return (*exc_raise_func)(exception_port, thread, task, exception, code, codeCnt);
#else
	extern kern_return_t catch_exception_raise(mach_port_t, mach_port_t, mach_port_t, exception_type_t, exception_data_t, mach_msg_type_number_t);
	return catch_exception_raise(exception_port, thread, task, exception, code, codeCnt);
#endif
}
