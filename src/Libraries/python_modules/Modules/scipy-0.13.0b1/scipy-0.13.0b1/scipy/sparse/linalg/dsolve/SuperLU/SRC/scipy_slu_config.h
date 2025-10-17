/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 28, 2022.
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

#ifndef SCIPY_SLU_CONFIG_H
#define SCIPY_SLU_CONFIG_H

#include <stdlib.h>

/*
 * Support routines
 */
void superlu_python_module_abort(char *msg);
void *superlu_python_module_malloc(size_t size);
void superlu_python_module_free(void *ptr);

#define USER_ABORT  superlu_python_module_abort
#define USER_MALLOC superlu_python_module_malloc
#define USER_FREE   superlu_python_module_free

#define SCIPY_FIX 1

/*
 * Fortran configuration
 */
#if defined(NO_APPEND_FORTRAN)
#if defined(UPPERCASE_FORTRAN)
#define UpCase 1
#else
#define NoChange 1
#endif
#else
#if defined(UPPERCASE_FORTRAN)
#error Uppercase and trailing slash in Fortran names not supported
#else
#define Add_ 1
#endif
#endif

#endif
