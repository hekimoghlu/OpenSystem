/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 13, 2024.
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
#ifndef SYS_MEMORYSTATUS_NOTIFY_H
#define SYS_MEMORYSTATUS_NOTIFY_H

#include <stdint.h>
#include <sys/proc.h>
#include <sys/param.h>

#if BSD_KERNEL_PRIVATE

#if VM_PRESSURE_EVENTS

extern vm_pressure_level_t memorystatus_vm_pressure_level;
extern _Atomic bool memorystatus_hwm_candidates;
extern unsigned int memorystatus_sustained_pressure_maximum_band;

boolean_t memorystatus_warn_process(const proc_t p, boolean_t is_active,
    boolean_t is_fatal, boolean_t exceeded);
int memorystatus_send_note(int event_code, void *data, uint32_t data_length);
void memorystatus_send_low_swap_note(void);
void consider_vm_pressure_events(void);
void memorystatus_notify_init(void);

#if CONFIG_MEMORYSTATUS

int memorystatus_low_mem_privileged_listener(uint32_t op_flags);
int memorystatus_send_pressure_note(int pid);
boolean_t memorystatus_is_foreground_locked(proc_t p);
boolean_t memorystatus_bg_pressure_eligible(proc_t p);
void memorystatus_proc_flags_unsafe(void * v, boolean_t *is_dirty,
    boolean_t *is_dirty_tracked, boolean_t *allow_idle_exit);
void memorystatus_broadcast_jetsam_pressure(
	vm_pressure_level_t pressure_level);

#endif /* CONFIG_MEMORYSTATUS */

#if DEBUG
#define VM_PRESSURE_DEBUG(cond, format, ...)      \
do {                                              \
if (cond) { printf(format, ##__VA_ARGS__); } \
} while(0)
#else
#define VM_PRESSURE_DEBUG(cond, format, ...)
#endif

#endif /* VM_PRESSURE_EVENTS */

#endif /* BSD_KERNEL_PRIVATE */

#endif /* SYS_MEMORYSTATUS_NOTIFY_H */
