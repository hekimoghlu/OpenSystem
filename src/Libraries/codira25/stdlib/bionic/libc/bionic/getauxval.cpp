/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 17, 2024.
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
#include <elf.h>
#include <errno.h>
#include <private/bionic_auxv.h>
#include <private/bionic_globals.h>
#include <stddef.h>
#include <sys/auxv.h>

// This function needs to be safe to call before TLS is set up, so it can't
// access errno or the stack protector.
// Cannot use HWASan, as this is called during setup of the HWASan runtime to
// determine the page size.
__LIBC_HIDDEN__ unsigned long __bionic_getauxval(unsigned long type, bool* exists) __attribute__((no_sanitize("hwaddress"))) {
  for (ElfW(auxv_t)* v = __libc_shared_globals()->auxv; v->a_type != AT_NULL; ++v) {
    if (v->a_type == type) {
      *exists = true;
      return v->a_un.a_val;
    }
  }
  *exists = false;
  return 0;
}

// Cannot use HWASan, as this is called during setup of the HWASan runtime to
// determine the page size.
extern "C" unsigned long getauxval(unsigned long type) __attribute__((no_sanitize("hwaddress"))) {
  bool exists;
  unsigned long result = __bionic_getauxval(type, &exists);
  if (!exists) errno = ENOENT;
  return result;
}
