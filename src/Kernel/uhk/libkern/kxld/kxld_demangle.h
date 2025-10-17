/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 10, 2022.
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
#ifndef _KXLD_DEMANGLE_H_
#define _KXLD_DEMANGLE_H_

#include <sys/types.h>

/* @function kxld_demangle
 *
 * @abstract Demangles c++ symbols.
 *
 * @param str           The C-string to be demangled.
 * @param buffer        A pointer to a character buffer for storing the result.
 *                      If NULL, a buffer will be malloc'd and stored here.
 *                      If the buffer is not large enough, it will be realloc'd.
 *
 * @param length        The length of the buffer.
 *
 * @result              If the input string could be demangled, it returns the
 *                      demangled string.  Otherwise, returns the input string.
 *
 */
const char * kxld_demangle(const char *str, char **buffer, size_t *length)
__attribute__((nonnull(1), visibility("hidden")));

#endif /* !_KXLD_DEMANGLE_H_ */
