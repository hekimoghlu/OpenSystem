/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 15, 2022.
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
extern void __cxa_finalize(void *);
extern void *__dso_handle;

__attribute__((destructor))
static void __on_dlclose(void) {
  __cxa_finalize(&__dso_handle);
}

/* Define a weak stub function here that will be overridden if the solib uses
 * emutls. The function needs to be a definition, not just a declaration,
 * because gold has a bug where it outputs weak+hidden symbols into the .dynsym
 * table. */
__attribute__((weak,visibility("hidden")))
void __emutls_unregister_key(void) {
}

/* Use a priority of 0 to run after any ordinary destructor function. The
 * priority setting moves the function towards the front of the .fini_array
 * section. */
__attribute__((destructor(0)))
static void __on_dlclose_late(void) {
  __emutls_unregister_key();
}

/* CRT_LEGACY_WORKAROUND should only be defined when building
 * this file as part of the platform's C library.
 *
 * The C library already defines a function named 'atexit()'
 * for backwards compatibility with older NDK-generated binaries.
 *
 * For newer ones, 'atexit' is actually embedded in the C
 * runtime objects that are linked into the final ELF
 * binary (shared library or executable), and will call
 * __cxa_atexit() in order to un-register any atexit()
 * handler when a library is unloaded.
 *
 * This function must be global *and* hidden. Only the
 * code inside the same ELF binary should be able to access it.
 */

#ifdef CRT_LEGACY_WORKAROUND
# include "__dso_handle.h"
#else
# include "__dso_handle_so.h"
# include "atexit.h"
#endif
#include "pthread_atfork.h"
