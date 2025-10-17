/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 3, 2021.
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
#include "compiler.h"
#include "nasmlib.h"

#ifdef HAVE_SYS_RESOURCE_H
# include <sys/resource.h>
#endif

#if defined(HAVE_GETRLIMIT) && defined(RLIMIT_STACK)

size_t nasm_get_stack_size_limit(void)
{
    struct rlimit rl;

    if (getrlimit(RLIMIT_STACK, &rl))
        return SIZE_MAX;

# ifdef RLIM_SAVED_MAX
    if (rl.rlim_cur == RLIM_SAVED_MAX)
        rl.rlim_cur = rl.rlim_max;
# endif

    if (
# ifdef RLIM_INFINITY
        rl.rlim_cur >= RLIM_INFINITY ||
# endif
# ifdef RLIM_SAVED_CUR
        rl.rlim_cur == RLIM_SAVED_CUR ||
# endif
# ifdef RLIM_SAVED_MAX
        rl.rlim_cur == RLIM_SAVED_MAX ||
# endif
        (size_t)rl.rlim_cur != rl.rlim_cur)
        return SIZE_MAX;

    return rl.rlim_cur;
}

#else

size_t nasm_get_stack_size_limit(void)
{
    return SIZE_MAX;
}

#endif
