/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 5, 2023.
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
#include <ffi.h>
#include <ffi_common.h>
#include <stdlib.h>
#include <stdio.h>

/* General debugging routines */

void ffi_stop_here(void)
{
  /* This function is only useful for debugging purposes.
     Place a breakpoint on ffi_stop_here to be notified of
     significant events. */
}

/* This function should only be called via the FFI_ASSERT() macro */

void ffi_assert(char *expr, char *file, int line)
{
  fprintf(stderr, "ASSERTION FAILURE: %s at %s:%d\n", expr, file, line);
  ffi_stop_here();
  abort();
}

/* Perform a sanity check on an ffi_type structure */

void ffi_type_test(ffi_type *a, char *file, int line)
{
  FFI_ASSERT_AT(a != NULL, file, line);

  FFI_ASSERT_AT(a->type <= FFI_TYPE_LAST, file, line);
  FFI_ASSERT_AT(a->type == FFI_TYPE_VOID || a->size > 0, file, line);
  FFI_ASSERT_AT(a->type == FFI_TYPE_VOID || a->alignment > 0, file, line);
  FFI_ASSERT_AT((a->type != FFI_TYPE_STRUCT && a->type != FFI_TYPE_COMPLEX)
		|| a->elements != NULL, file, line);
  FFI_ASSERT_AT(a->type != FFI_TYPE_COMPLEX
		|| (a->elements != NULL
		    && a->elements[0] != NULL && a->elements[1] == NULL),
		file, line);

}

