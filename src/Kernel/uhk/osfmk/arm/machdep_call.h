/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 26, 2025.
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

typedef union {
	kern_return_t           (*args_0)(void);
	kern_return_t           (*args_1)(vm_address_t);
	kern_return_t           (*args_2)(vm_address_t, vm_address_t);
	kern_return_t           (*args_3)(vm_address_t, vm_address_t, vm_address_t);
	kern_return_t           (*args_4)(vm_address_t, vm_address_t, vm_address_t, vm_address_t);
	kern_return_t           (*args_var)(vm_address_t, ...);
} machdep_call_routine_t;

#define MACHDEP_CALL_ROUTINE(func, args) \
	{ { .args_ ## args = func }, args }

typedef struct {
	machdep_call_routine_t      routine;
	int                         nargs;
} machdep_call_t;

extern const machdep_call_t             machdep_call_table[];
extern int                      machdep_call_count;

extern vm_address_t             thread_get_cthread_self(void);
extern kern_return_t            thread_set_cthread_self(vm_address_t);

// Read and write raw TPIDRURO / TPIDRRO_EL0
uintptr_t                       get_tpidrro(void);
void                            set_tpidrro(uintptr_t);
