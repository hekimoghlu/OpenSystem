/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 21, 2023.
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
#ifndef _STDLIB_H
#error "Never include this file directly; instead, include <stdlib.h>"
#endif

#if defined(__BIONIC_FORTIFY)

/* PATH_MAX is unavailable without polluting the namespace, but it's always 4096 on Linux */
#define __PATH_MAX 4096

char* _Nullable realpath(const char* _Nonnull path, char* _Nullable resolved)
        __clang_error_if(!path, "'realpath': NULL path is never correct; flipped arguments?")
        __clang_error_if(__bos_unevaluated_lt(__bos(resolved), __PATH_MAX),
                         "'realpath' output parameter must be NULL or a pointer to a buffer "
                         "with >= PATH_MAX bytes");

/* No need for a definition; the only issues we can catch are at compile-time. */

#undef __PATH_MAX
#endif /* defined(__BIONIC_FORTIFY) */
