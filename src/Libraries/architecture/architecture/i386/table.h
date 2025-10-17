/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 4, 2022.
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
/*
 * Copyright (c) 1992 NeXT Computer, Inc.
 *
 * Intel386 Family:	Descriptor tables.
 *
 * HISTORY
 *
 * 30 March 1992 ? at NeXT
 *	Created.
 */

#ifndef _ARCH_I386_TABLE_H_
#define _ARCH_I386_TABLE_H_

#include <architecture/i386/desc.h>
#include <architecture/i386/tss.h>

/*
 * A totally generic descriptor
 * table entry.
 */

typedef union dt_entry {
    code_desc_t		code;
    data_desc_t		data;
    ldt_desc_t		ldt;
    tss_desc_t		task_state;
    call_gate_t		call_gate;
    trap_gate_t		trap_gate;
    intr_gate_t		intr_gate;
    task_gate_t		task_gate;
} dt_entry_t;

#define DESC_TBL_MAX	8192

/*
 * Global descriptor table.
 */

typedef union gdt_entry {
    code_desc_t		code;
    data_desc_t		data;
    ldt_desc_t		ldt;
    call_gate_t		call_gate;
    task_gate_t		task_gate;
    tss_desc_t		task_state;
} gdt_entry_t;

typedef gdt_entry_t	gdt_t;

/*
 * Interrupt descriptor table.
 */

typedef union idt_entry {
    trap_gate_t		trap_gate;
    intr_gate_t		intr_gate;
    task_gate_t		task_gate;
} idt_entry_t;

typedef idt_entry_t	idt_t;

/*
 * Local descriptor table.
 */

typedef union ldt_entry {
    code_desc_t		code;
    data_desc_t		data;
    call_gate_t		call_gate;
    task_gate_t		task_gate;
} ldt_entry_t;

typedef ldt_entry_t	ldt_t;

#endif	/* _ARCH_I386_TABLE_H_ */
