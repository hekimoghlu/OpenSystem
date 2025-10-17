/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 19, 2024.
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
#ifndef PAS_DARWIN_SPI_H
#define PAS_DARWIN_SPI_H

#include "pas_utils.h"
#include <pthread.h>

#if PAS_OS(DARWIN)
#if defined(__has_include) && __has_include(<pthread/private.h>)

// FIXME: rdar://140431798 Remove PAS_{BEGIN/END}_EXTERN_C when WebKit does not need to support the platform versions anymore.
#if !((PAS_PLATFORM(MAC) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 150400) \
    || (PAS_PLATFORM(IOS) && __IPHONE_OS_VERSION_MAX_ALLOWED >= 180400) \
    || (PAS_PLATFORM(APPLETV) && __TV_OS_VERSION_MAX_ALLOWED >= 180400) \
    || (PAS_PLATFORM(WATCHOS) && __WATCH_OS_VERSION_MAX_ALLOWED >= 110400) \
    || (PAS_PLATFORM(VISION) && __VISION_OS_VERSION_MAX_ALLOWED >= 20040))
PAS_BEGIN_EXTERN_C;
#endif
#include <pthread/private.h>
#if !((PAS_PLATFORM(MAC) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 150400) \
    || (PAS_PLATFORM(IOS) && __IPHONE_OS_VERSION_MAX_ALLOWED >= 180400) \
    || (PAS_PLATFORM(APPLETV) && __TV_OS_VERSION_MAX_ALLOWED >= 180400) \
    || (PAS_PLATFORM(WATCHOS) && __WATCH_OS_VERSION_MAX_ALLOWED >= 110400) \
    || (PAS_PLATFORM(VISION) && __VISION_OS_VERSION_MAX_ALLOWED >= 20040))
PAS_END_EXTERN_C;
#endif

#define PAS_HAVE_PTHREAD_PRIVATE 1
#else
PAS_BEGIN_EXTERN_C;
int pthread_self_is_exiting_np(void);
PAS_END_EXTERN_C;
#define PAS_HAVE_PTHREAD_PRIVATE 0
#endif

PAS_BEGIN_EXTERN_C;

/* From OSS libmalloc stack_logging.h
   https://github.com/apple-oss-distributions/libmalloc/blob/main/private/stack_logging.h */
/*********    MallocStackLogging permanant SPIs  ************/

#define pas_stack_logging_type_free                           0
#define pas_stack_logging_type_generic                        1    /* anything that is not allocation/deallocation */
#define pas_stack_logging_type_alloc                          2    /* malloc, realloc, etc... */
#define pas_stack_logging_type_dealloc                        4    /* free, realloc, etc... */
#define pas_stack_logging_type_vm_allocate                   16    /* vm_allocate or mmap */
#define pas_stack_logging_type_vm_deallocate                 32    /* vm_deallocate or munmap */
#define pas_stack_logging_type_mapped_file_or_shared_mem    128

// The valid flags include those from VM_FLAGS_ALIAS_MASK, which give the user_tag of allocated VM regions.
#define pas_stack_logging_valid_type_flags ( \
pas_stack_logging_type_generic | \
pas_stack_logging_type_alloc | \
pas_stack_logging_type_dealloc | \
pas_stack_logging_type_vm_allocate | \
pas_stack_logging_type_vm_deallocate | \
pas_stack_logging_type_mapped_file_or_shared_mem | \
VM_FLAGS_ALIAS_MASK);

// Following flags are absorbed by stack_logging_log_stack()
#define pas_stack_logging_flag_zone        8    /* NSZoneMalloc, etc... */
#define pas_stack_logging_flag_cleared    64    /* for NewEmptyHandle */

typedef void(malloc_logger_t)(uint32_t type,
                              uintptr_t arg1,
                              uintptr_t arg2,
                              uintptr_t arg3,
                              uintptr_t result,
                              uint32_t num_hot_frames_to_skip);
// FIXME: Workaround for rdar://119319825
#if !defined(__swift__)
extern malloc_logger_t* malloc_logger;
#endif

PAS_END_EXTERN_C;

#endif /* PAS_OS(DARWIN) */

#endif /* PAS_DARWIN_SPI_H */
