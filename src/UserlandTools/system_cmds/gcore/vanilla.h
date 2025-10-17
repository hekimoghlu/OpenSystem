/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 10, 2023.
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
#include "vm.h"
#include "notes.h"

#ifndef _VANILLA_H
#define _VANILLA_H

struct proc_bsdinfo;

extern void validate_core_header(const native_mach_header_t *, off_t);
extern int coredump(task_t, int, const struct proc_bsdinfo *);
extern int coredump_pwrite(task_t, int, struct regionhead *, const uuid_t, mach_vm_offset_t, mach_vm_offset_t, const struct task_crashinfo_note_data *, const struct region_infos_note_data *);
extern int coredump_stream(task_t, int, struct regionhead *);
extern struct regionhead *coredump_prepare(task_t, uuid_t);
extern bool can_remove_region(struct region *r);
#endif /* _VANILLA_H */
