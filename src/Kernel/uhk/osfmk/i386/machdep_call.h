/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 15, 2024.
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
 * Machine dependent kernel call table defines.
 *
 * HISTORY
 *
 * 17 June 1992 ? at NeXT
 *	Created.
 */

/* The maximum number of fixed arguments any machdep routine takes. */
#define MACHDEP_MAX_ARGS 4

typedef union {
	kern_return_t           (*args_0)(void);
	kern_return_t           (*args_1)(uint32_t);
	kern_return_t           (*args64_1)(uint64_t);
	kern_return_t           (*args_2)(uint32_t, uint32_t);
	kern_return_t           (*args64_2)(uint64_t, uint64_t);
	kern_return_t           (*args_3)(uint32_t, uint32_t, uint32_t);
	kern_return_t           (*args64_3)(uint64_t, uint64_t, uint64_t);
	kern_return_t           (*args_4)(uint32_t, uint32_t, uint32_t, uint32_t);
	kern_return_t           (*args_var)(uint32_t, ...);
	int                     (*args_bsd_3)(uint32_t *, uint32_t,
	    uint32_t, uint32_t);
	int                     (*args64_bsd_3)(uint32_t *, uint64_t,
	    uint64_t, uint64_t);
} machdep_call_routine_t;

#define MACHDEP_CALL_ROUTINE(func, args)        \
	{ { .args_ ## args = func }, args, 0 }

#define MACHDEP_CALL_ROUTINE64(func, args)      \
	{ { .args64_ ## args = func }, args, 0 }

#define MACHDEP_BSD_CALL_ROUTINE(func, args)    \
	{ { .args_bsd_ ## args = func }, args, 1 }

#define MACHDEP_BSD_CALL_ROUTINE64(func, args)    \
	{ { .args64_bsd_ ## args = func }, args, 1 }

typedef struct {
	machdep_call_routine_t      routine;
	int                         nargs;
	int                         bsd_style;
} machdep_call_t;

extern const machdep_call_t             machdep_call_table[];
extern const machdep_call_t             machdep_call_table64[];

extern int                      machdep_call_count;

#if HYPERVISOR
extern kern_return_t            hv_task_trap(uint64_t, uint64_t);
extern kern_return_t            hv_thread_trap(uint64_t, uint64_t);
#endif

extern kern_return_t            thread_fast_set_cthread_self(uint32_t);
extern kern_return_t            thread_fast_set_cthread_self64(uint64_t);
extern kern_return_t            thread_set_user_ldt(uint32_t, uint32_t, uint32_t);

extern int              i386_set_ldt(uint32_t *, uint32_t, uint32_t, uint32_t);
extern int              i386_get_ldt(uint32_t *, uint32_t, uint32_t, uint32_t);
extern int              i386_set_ldt64(uint32_t *, uint64_t, uint64_t, uint64_t);
extern int              i386_get_ldt64(uint32_t *, uint64_t, uint64_t, uint64_t);

extern void                     machdep_syscall(x86_saved_state_t *);
extern void                     machdep_syscall64(x86_saved_state_t *);
