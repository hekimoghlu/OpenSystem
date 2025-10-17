/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 10, 2023.
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
#ifndef _MALLOC_UNDERSCORE_PLATFORM_H
#define _MALLOC_UNDERSCORE_PLATFORM_H

#include <malloc/_ptrcheck.h>
__ptrcheck_abi_assume_single()

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include <sys/cdefs.h>
#ifndef __DARWIN_EXTSN
#define __DARWIN_EXTSN(x)
#endif /* __DARWIN_EXTSN */

#if __has_include(<mach/boolean.h>)
#include <mach/boolean.h>
#else
typedef int boolean_t;
#endif /* __has_include(<mach/boolean.h>) */

#if __has_include(<mach/kern_return.h>)
#include <mach/kern_return.h>
#else
typedef int kern_return_t;
#endif /* __has_include(<mach/kern_return.h>) */

#if __has_include(<mach/port.h>)
#include <mach/port.h>
#else
typedef void * __single mach_port_t;
#endif /* __has_include(<mach/port.h>) */

#if __has_include(<mach/mach_types.h>)
#include <mach/mach_types.h>
#else
typedef mach_port_t task_t;
#endif /* __has_include(<mach/mach_types.h>) */

#if __has_include(<mach/vm_types.h>)
#include <mach/vm_types.h>
#else
typedef uint64_t mach_vm_address_t;
typedef uint64_t mach_vm_offset_t;
typedef uint64_t mach_vm_size_t;
typedef uintptr_t vm_offset_t;
typedef vm_offset_t vm_address_t;
typedef uintptr_t vm_size_t;
#endif /* __has_include(<mach/vm_types.h>) */

#if __has_include(<malloc/_malloc_type.h>) && !defined(_ANSI_SOURCE) && (!defined(_POSIX_C_SOURCE) || defined(_DARWIN_C_SOURCE))
#include <malloc/_malloc_type.h>
#else
#define _MALLOC_TYPED(override, type_param_pos)
#endif

#endif /* _MALLOC_UNDERSCORE_PLATFORM_H */
