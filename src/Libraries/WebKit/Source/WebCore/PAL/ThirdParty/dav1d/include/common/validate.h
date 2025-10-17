/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 10, 2024.
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
#ifndef DAV1D_COMMON_VALIDATE_H
#define DAV1D_COMMON_VALIDATE_H

#include <stdio.h>
#include <stdlib.h>

#if defined(NDEBUG)
#define debug_abort()
#else
#define debug_abort abort
#endif

#define validate_input_or_ret_with_msg(x, r, ...) \
    if (!(x)) { \
        fprintf(stderr, "Input validation check \'%s\' failed in %s!\n", \
                #x, __func__); \
        fprintf(stderr, __VA_ARGS__); \
        debug_abort(); \
        return r; \
    }

#define validate_input_or_ret(x, r) \
    if (!(x)) { \
        fprintf(stderr, "Input validation check \'%s\' failed in %s!\n", \
                #x, __func__); \
        debug_abort(); \
        return r; \
    }

#define validate_input(x) validate_input_or_ret(x, )

#endif /* DAV1D_COMMON_VALIDATE_H */
