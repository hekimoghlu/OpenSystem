/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 11, 2023.
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
#ifndef __IOKIT_SYSTEM_H
#define __IOKIT_SYSTEM_H

#ifndef __STDC_LIMIT_MACROS
#define __STDC_LIMIT_MACROS
#endif

#include <sys/cdefs.h>
#include <kern/thread.h>

#include <stdarg.h>
#include <stdint.h>
#include <string.h>

#include <IOKit/assert.h>  /* Must be before other includes of kern/assert.h */

#include <kern/debug.h>
#include <kern/task.h>
#include <kern/sched_prim.h>
#include <kern/locks.h>
#include <kern/queue.h>
#include <kern/ipc_mig.h>
#ifndef MACH_KERNEL_PRIVATE
#include <libkern/libkern.h>
#endif

#ifdef  KERNEL_PRIVATE
#include <kern/kalloc.h>
#endif /* KERNEL_PRIVATE */

__BEGIN_DECLS

#include <mach/mach_types.h>
#include <mach/mach_interface.h>
#include <mach/memory_object_types.h>

#include <kern/kern_types.h>

#ifdef  KERNEL_PRIVATE
#include <vm/pmap.h>
#include <vm/vm_map.h>
#include <vm/vm_kern.h>
#endif /* KERNEL_PRIVATE */

#ifndef _MISC_PROTOS_H_
extern void     _doprnt( const char *format, va_list *arg,
    void (*lputc)(char), int radix );
#endif

__END_DECLS

#endif /* !__IOKIT_SYSTEM_H */
