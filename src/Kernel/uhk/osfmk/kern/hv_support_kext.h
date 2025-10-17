/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 21, 2024.
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
#ifndef _KERN_HV_SUPPORT_KEXT_H_
#define _KERN_HV_SUPPORT_KEXT_H_

#if defined(__cplusplus)
extern "C" {
#endif

#include <stdint.h>
#include <kern/kern_types.h>
#include <mach/kern_return.h>
#include <kern/hv_io_notifier.h>

typedef enum {
	HV_DEBUG_STATE
} hv_volatile_state_t;

typedef enum {
	HV_TASK_TRAP = 0,
	HV_THREAD_TRAP = 1
} hv_trap_type_t;

typedef kern_return_t (*hv_trap_t) (void *target, uint64_t arg);

typedef struct  {
	const hv_trap_t *traps;
	unsigned trap_count;
} hv_trap_table_t;

typedef struct {
	void (*dispatch)(void *vcpu);
	void (*preempt)(void *vcpu);
	void (*suspend)(void);
	void (*thread_destroy)(void *vcpu);
	void (*task_destroy)(void *vm);
	void (*volatile_state)(void *vcpu, int state);
#define HV_CALLBACKS_RESUME_DEFINED 1
	void (*resume)(void);
	void (*memory_pressure)(void);
} hv_callbacks_t;

extern hv_callbacks_t hv_callbacks;
extern int hv_support_available;

extern void hv_support_init(void);
extern int hv_get_support(void);
extern void hv_set_task_target(void *target);
extern void hv_set_thread_target(void *target);
extern void *hv_get_task_target(void);
extern void *hv_get_thread_target(void);
extern int hv_get_volatile_state(hv_volatile_state_t state);
extern kern_return_t hv_set_traps(hv_trap_type_t trap_type,
    const hv_trap_t *traps, unsigned trap_count);
extern void hv_release_traps(hv_trap_type_t trap_type);
extern kern_return_t hv_set_callbacks(hv_callbacks_t callbacks);
extern void hv_release_callbacks(void);
extern void hv_suspend(void);
extern void hv_resume(void);
extern kern_return_t hv_task_trap(uint64_t index, uint64_t arg);
extern kern_return_t hv_thread_trap(uint64_t index, uint64_t arg);
extern boolean_t hv_ast_pending(void);

extern void hv_trace_guest_enter(uint32_t vcpu_id, uint64_t *vcpu_regs);
extern void hv_trace_guest_exit(uint32_t vcpu_id, uint64_t *vcpu_regs,
    uint32_t reason);
extern void hv_trace_guest_error(uint32_t vcpu_id, uint64_t *vcpu_regs,
    uint32_t failure, uint32_t error);

#if defined(__cplusplus)
}
#endif

#endif /* _KERN_HV_SUPPORT_KEXT_H_ */
