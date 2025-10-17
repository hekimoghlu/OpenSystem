/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 5, 2025.
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
#ifndef   __COPYOUT_SHIM_X86_64_H__
#define   __COPYOUT_SHIM_X86_64_H__
#ifdef KERNEL_PRIVATE

// Osfmk includes libsa/types.h which causes massive conflicts
// with sys/types.h
#if defined (__i386__) || defined(__x86_64__)
#include "i386/types.h"
#elif defined (__arm__) || defined (__arm64__)
//XXX when ready to turn on for arm: #include "arm/types.h"
#error ARM/ARM64 not supported
#else
#error architecture not supported
#endif

#include <mach/mach_types.h>

#define CO_SRC_NORMAL 1       //copyout() called
#define CO_SRC_MSG    (1<<1)    //copyoutmsg() called
#define CO_SRC_PHYS   (1<<2)    //copyio(COPYOUTPHYS,...) called

typedef void (*copyout_shim_fn_t)(const void *, user_addr_t, vm_size_t, unsigned co_src);

#ifdef MACH_KERNEL_PRIVATE
#if(DEVELOPMENT || DEBUG) && (COPYOUT_SHIM > 0)

extern copyout_shim_fn_t copyout_shim_fn;
extern unsigned co_src_flags;

// void call_copyout_shim(const void *kernel_addr,user_addr_t user_addr,vm_size_t nbytes,int copy_type,int copyout_flavors);

#define CALL_COPYOUT_SHIM_NRML(ka, ua, nb) \
    if(copyout_shim_fn && (co_src_flags & CO_SRC_NORMAL)) {copyout_shim_fn(ka,ua,nb,CO_SRC_NORMAL); }

#define CALL_COPYOUT_SHIM_MSG(ka, ua, nb) \
    if(copyout_shim_fn && (co_src_flags & CO_SRC_MSG)){copyout_shim_fn(ka,ua,nb,CO_SRC_MSG); }

#define CALL_COPYOUT_SHIM_PHYS(ka, ua, nb) \
    if(copyout_shim_fn && (co_src_flags & CO_SRC_PHYS)){copyout_shim_fn(ka,ua,nb,CO_SRC_PHYS); }

#else
//Make these calls disappear if we're RELEASE or if COPYOUT_SHIM didn't get built
#define CALL_COPYOUT_SHIM_NRML(ka, ua, nb)
#define CALL_COPYOUT_SHIM_MSG(ka, ua, nb)
#define CALL_COPYOUT_SHIM_PHYS(ka, ua, nb)
#endif /* (DEVELOPMENT || DEBUG) && (COPYOUT_SHIM > 0) */
#endif /* MACH_KERNEL_PRIVATE */


kern_return_t
register_copyout_shim(copyout_shim_fn_t copyout_shim_fn, unsigned co_src_flags);


#define unregister_copyout_shim() register_copyout_shim(NULL,0)

void *
cos_kernel_unslide(const void *);

void *
cos_kernel_reslide(const void *);

#endif /* KERNEL_PRIVATE */
#endif /* __COPYOUT_SHIM_X86_64_H__ */
