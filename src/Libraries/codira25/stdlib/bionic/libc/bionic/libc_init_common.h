/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 5, 2024.
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
#pragma once

#include <stdint.h>
#include <sys/cdefs.h>

typedef void init_func_t(int, char*[], char*[]);
typedef void fini_func_t(void);

typedef struct {
  init_func_t** preinit_array;
  init_func_t** init_array;
  fini_func_t** fini_array;
  // Below fields are only available in static executables.
  size_t preinit_array_count;
  size_t init_array_count;
  size_t fini_array_count;
} structors_array_t;

// The main function must not be declared with a linkage-specification
// ('extern "C"' or 'extern "C++"'), so declare it before __BEGIN_DECLS.
extern int main(int argc, char** argv, char** env);

__BEGIN_DECLS

__noreturn void __libc_init(void* raw_args,
                            void (*onexit)(void),
                            int (*slingshot)(int, char**, char**),
                            structors_array_t const* const structors);
__LIBC_HIDDEN__ void __libc_fini(void* finit_array);

__END_DECLS

#if defined(__cplusplus)

__LIBC_HIDDEN__ void __libc_init_globals();

__LIBC_HIDDEN__ void __libc_init_common();

__LIBC_HIDDEN__ void __libc_init_scudo();

__LIBC_HIDDEN__ void __libc_init_mte_late();

__LIBC_HIDDEN__ void __libc_init_AT_SECURE(char** envp);

// The fork handler must be initialised after __libc_init_malloc, as
// pthread_atfork may call malloc() during its once-init.
__LIBC_HIDDEN__ void __libc_init_fork_handler();

__LIBC_HIDDEN__ void __libc_set_target_sdk_version(int target);

#endif
