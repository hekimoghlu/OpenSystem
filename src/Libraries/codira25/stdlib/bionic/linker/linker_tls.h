/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 10, 2023.
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

#include <stdlib.h>

#include "private/bionic_elf_tls.h"

struct TlsModule;
struct soinfo;

void linker_setup_exe_static_tls(const char* progname);
void linker_finalize_static_tls();

void register_soinfo_tls(soinfo* si);
void unregister_soinfo_tls(soinfo* si);

const TlsModule& get_tls_module(size_t module_id);

typedef size_t TlsDescResolverFunc(size_t);

struct TlsDescriptor {
#if defined(__arm__)
  size_t arg;
  TlsDescResolverFunc* func;
#else
  TlsDescResolverFunc* func;
  size_t arg;
#endif
};

struct TlsDynamicResolverArg {
  size_t generation;
  TlsIndex index;
};

__LIBC_HIDDEN__ extern "C" size_t tlsdesc_resolver_static(size_t);
__LIBC_HIDDEN__ extern "C" size_t tlsdesc_resolver_dynamic(size_t);
__LIBC_HIDDEN__ extern "C" size_t tlsdesc_resolver_unresolved_weak(size_t);
