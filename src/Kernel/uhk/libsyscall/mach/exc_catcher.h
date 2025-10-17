/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 2, 2024.
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
#ifndef __EXC_CATCHER_H
#define __EXC_CATCHER_H

#include "_libkernel_init.h"

typedef kern_return_t (*_libkernel_exc_raise_func_t)(mach_port_t,
    mach_port_t,
    mach_port_t,
    exception_type_t,
    exception_data_t,
    mach_msg_type_number_t);

typedef kern_return_t (*_libkernel_exc_raise_state_func_t)(mach_port_t,
    exception_type_t,
    exception_data_t,
    mach_msg_type_number_t,
    int *,
    thread_state_t,
    mach_msg_type_number_t,
    thread_state_t,
    mach_msg_type_number_t *);

typedef kern_return_t (*_libkernel_exec_raise_state_identity_t)(mach_port_t,
    mach_port_t, mach_port_t,
    exception_type_t,
    exception_data_t,
    mach_msg_type_number_t,
    int *, thread_state_t,
    mach_msg_type_number_t,
    thread_state_t,
    mach_msg_type_number_t *);

#define RTLD_DEFAULT    ((void *) -2)
extern void* (*_dlsym)(void*, const char*);

#endif // __EXC_CATCHER_H
