/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 16, 2024.
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

#include <_simple.h>

#include <platform/string.h>
#include <platform/compat.h>

const char *
_simple_getenv(const char *envp[], const char *var) {
    const char **p;
    size_t var_len;

    var_len = strlen(var);

    for (p = envp; p && *p; p++) {
        size_t p_len = strlen(*p);

        if (p_len >= var_len &&
            memcmp(*p, var, var_len) == 0 &&
            (*p)[var_len] == '=') {
            return &(*p)[var_len + 1];
        }
    }

    return NULL;
}
