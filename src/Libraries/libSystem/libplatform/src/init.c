/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 6, 2025.
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
#include <TargetConditionals.h>
#include <stdlib.h>
#include <platform/string.h>
#include <_libkernel_init.h>

struct ProgramVars; /* forward reference */

extern void _simple_asl_init(const char *envp[], const struct ProgramVars *vars);
extern void __pfz_setup(const char *apple[]);

#if !VARIANT_STATIC
static const struct _libkernel_string_functions _platform_string_functions = {
	.version = 1,
	.bzero = _platform_bzero,
	.memchr = _platform_memchr,
	.memcmp = _platform_memcmp,
	.memmove = _platform_memmove,
	.memccpy = _platform_memccpy,
	.memset = _platform_memset,
	.strchr = _platform_strchr,
	.strcmp = _platform_strcmp,
	.strcpy = _platform_strcpy,
	.strlcat = _platform_strlcat,
	.strlcpy = _platform_strlcpy,
	.strlen = _platform_strlen,
	.strncmp = _platform_strncmp,
	.strncpy = _platform_strncpy,
	.strnlen = _platform_strnlen,
	.strstr = _platform_strstr,
};
#endif

void
__libplatform_init(void *future_use __unused, const char *envp[],
		const char *apple[], const struct ProgramVars *vars)
{
    /* In the Simulator, we just provide _simple for dyld */
#if !TARGET_OS_SIMULATOR
    __pfz_setup(apple);
#endif
#if !TARGET_OS_DRIVERKIT
    _simple_asl_init(envp, vars);
#endif

#if !VARIANT_STATIC
    __libkernel_platform_init(&_platform_string_functions);
#endif
}
