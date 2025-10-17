/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 30, 2022.
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
#include "ffitest.h"

void doit(ffi_cif *cif, void *rvalue, void **avalue, void *closure)
{
  (void)cif;
  (void)avalue;
  *(void **)rvalue = closure;
}

typedef void * (*FN)(void);

int main()
{
  ffi_cif cif;
  ffi_go_closure cl;
  void *result;

  CHECK(ffi_prep_cif(&cif, ABI_NUM, 0, &ffi_type_pointer, NULL) == FFI_OK);
  CHECK(ffi_prep_go_closure(&cl, &cif, doit) == FFI_OK);

  ffi_call_go(&cif, FFI_FN(*(FN *)&cl), &result, NULL, &cl);

  CHECK(result == &cl);

  exit(0);
}
