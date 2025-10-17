/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 28, 2022.
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
#include "cc_internal.h"

//cc_abort() is implemented to comply with by FIPS 140-2, when DRBG produces
//two equal consecutive blocks.

#if !CC_PROVIDES_ABORT

#error "This environment does not provide an abort()/panic()-like function"

#elif CC_KERNEL

#include <kern/debug.h>
void
cc_abort(const char * msg)
{
	panic("%s", msg);
}

#elif CC_USE_L4

#include <sys/panic.h>
#include <stdarg.h>
void
cc_abort(const char * msg)
{
	sys_panic(msg);
}

#elif CC_RTKIT

#include <RTK_platform.h>
void
cc_abort(const char * msg)
{
	RTK_abort("%s", msg);
}

#else

#if CC_BUILT_FOR_TESTING
void (*cc_abort_mock)(const char *msg);
#endif

#include <stdlib.h>
void
cc_abort(CC_UNUSED const char *msg)
{
#if CC_BUILT_FOR_TESTING
	if (cc_abort_mock) {
		cc_abort_mock(msg);
	}
#endif

	abort();
}

#endif

