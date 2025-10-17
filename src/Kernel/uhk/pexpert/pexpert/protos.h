/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 14, 2025.
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
#ifndef _PEXPERT_PROTOS_H_
#define _PEXPERT_PROTOS_H_

#ifdef PEXPERT_KERNEL_PRIVATE


#include <mach/mach_types.h>
#include <mach/vm_types.h>
#include <mach/boolean.h>
#include <stdarg.h>
#include <string.h>
#include <kern/assert.h>

#include <pexpert/machine/protos.h>

//------------------------------------------------------------------------
// from ppc/misc_protos.h
extern void printf(const char *fmt, ...) __printflike(1, 2);

extern void interrupt_enable(void);
extern void interrupt_disable(void);
#define bcopy_nc bcopy

//------------------------------------------------------------------------
//from kern/misc_protos.h
extern void
_doprnt(
	const char     *fmt,
	va_list                 *argp,
	void                    (*putc)(char),
	int                     radix);

extern void
_doprnt_log(
	const char     *fmt,
	va_list                 *argp,
	void                    (*putc)(char),
	int                     radix);

//------------------------------------------------------------------------
// ??
//typedef int kern_return_t;
void Debugger(const char *message);

#include <kern/cpu_number.h>
#include <kern/cpu_data.h>

//------------------------------------------------------------------------
// from kgdb/kgdb_defs.h
#define kgdb_printf printf

#include <mach/machine/vm_types.h>
#include <device/device_types.h>
#include <kern/kalloc.h>

//------------------------------------------------------------------------

// from iokit/IOStartIOKit.cpp
extern void InitIOKit(void *dtTop);
extern void ConfigureIOKit(void);
extern void StartIOKitMatching(void);

// from iokit/Families/IOFramebuffer.cpp
extern unsigned char appleClut8[256 * 3];


#endif /* PEXPERT_KERNEL_PRIVATE */

#endif /* _PEXPERT_PROTOS_H_ */
