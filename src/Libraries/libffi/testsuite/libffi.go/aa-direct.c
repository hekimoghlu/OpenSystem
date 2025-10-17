/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 23, 2023.
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
#include "static-chain.h"

#if defined(__GNUC__) && !defined(__clang__) && defined(STATIC_CHAIN_REG)

#include "ffitest.h"

/* Blatent assumption here that the prologue doesn't clobber the
   static chain for trivial functions.  If this is not true, don't
   define STATIC_CHAIN_REG, and we'll test what we can via other tests.  */
void *doit(void)
{
  register void *chain __asm__(STATIC_CHAIN_REG);
  return chain;
}

int main()
{
  ffi_cif cif;
  void *result;

  CHECK(ffi_prep_cif(&cif, ABI_NUM, 0, &ffi_type_pointer, NULL) == FFI_OK);

  ffi_call_go(&cif, FFI_FN(doit), &result, NULL, &result);

  CHECK(result == &result);

  return 0;
}

#else /* UNSUPPORTED */
int main() { return 0; }
#endif
