/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 24, 2022.
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
#ifndef _OBJC_VM_H
#define _OBJC_VM_H

/* 
 * WARNING  DANGER  HAZARD  BEWARE  EEK
 * 
 * Everything in this file is for Apple Internal use only.
 * These will change in arbitrary OS updates and in unpredictable ways.
 * When your program breaks, you get to keep both pieces.
 */

/*
 * objc-vm.h: defines PAGE_SIZE, PAGE_MIN/MAX_SIZE and PAGE_MAX_SHIFT
 */

// N.B. This file must be usable FROM ASSEMBLY SOURCE FILES

#include <TargetConditionals.h>

#if __has_include(<mach/vm_param.h>)
#  include <mach/vm_param.h>

#  define OBJC_VM_MAX_ADDRESS    MACH_VM_MAX_ADDRESS
#elif __arm64__
#  define PAGE_SIZE       16384
#  define PAGE_MIN_SIZE   16384
#  define PAGE_MAX_SIZE   16384
#  define PAGE_MAX_SHIFT  14
#if TARGET_OS_EXCLAVEKIT
#  define OBJC_VM_MAX_ADDRESS  0x0000000ffffffff8ULL
#else
#  define OBJC_VM_MAX_ADDRESS  0x00007ffffffffff8ULL
#endif
#else
#  error Unknown platform - please define PAGE_SIZE et al.
#endif

#endif // _OBJC_VM_H
