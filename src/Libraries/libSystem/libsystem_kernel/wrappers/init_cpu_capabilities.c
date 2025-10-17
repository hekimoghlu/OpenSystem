/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 25, 2021.
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
#define __APPLE_API_PRIVATE
#include <machine/cpu_capabilities.h>
#undef  __APPLE_API_PRIVATE

#if defined(__i386__) || defined(__x86_64__)

/* Initialize the "_cpu_capabilities" vector on x86 processors. */

int _cpu_has_altivec = 0;     // DEPRECATED
int _cpu_capabilities = 0;

void
_init_cpu_capabilities( void )
{
	_cpu_capabilities = (int)_get_cpu_capabilities();
}

#elif defined(__arm__) || defined(__arm64__)

extern uint64_t _get_cpu_capabilities(void);

int _cpu_capabilities = 0;
int _cpu_has_altivec = 0;               // DEPRECATED: use _cpu_capabilities instead

void
_init_cpu_capabilities( void )
{
}

#endif
