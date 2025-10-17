/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 4, 2023.
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
#ifndef _KERN_SFI_H_
#define _KERN_SFI_H_

#include <stdint.h>
#include <mach/mach_types.h>
#include <mach/kern_return.h>
#include <mach/sfi_class.h>
#include <kern/ast.h>
#include <kern/kern_types.h>
#include <kern/ledger.h>

#if KERNEL_PRIVATE
#if !XNU_KERNEL_PRIVATE
#error "This file is for internal use and will be deleted in future versions of the SDK."
#endif /* !XNU_KERNEL_PRIVATE */
#endif /* KERNEL_PRIVATE */

#if XNU_KERNEL_PRIVATE
extern void sfi_init(void);
extern sfi_class_id_t sfi_get_ledger_alias_for_class(sfi_class_id_t class_id);

kern_return_t sfi_set_window(uint64_t window_usecs);
kern_return_t sfi_window_cancel(void);
kern_return_t sfi_get_window(uint64_t *window_usecs);

kern_return_t sfi_set_class_offtime(sfi_class_id_t class_id, uint64_t offtime_usecs);
kern_return_t sfi_class_offtime_cancel(sfi_class_id_t class_id);
kern_return_t sfi_get_class_offtime(sfi_class_id_t class_id, uint64_t *offtime_usecs);

#ifdef MACH_KERNEL_PRIVATE
/*
 * Classifying a thread requires no special locks to be held (although attribute
 * changes that cause an inconsistent snapshot may cause a spurious AST). Final
 * evaluation will happen at the AST boundary with the thread locked. If possible,
 *
 */
sfi_class_id_t sfi_thread_classify(thread_t thread);
sfi_class_id_t sfi_processor_active_thread_classify(processor_t processor);
ast_t sfi_thread_needs_ast(thread_t thread, sfi_class_id_t *out_class /* optional */);
ast_t sfi_processor_needs_ast(processor_t processor);

void sfi_ast(thread_t thread);
void sfi_reevaluate(thread_t thread);
kern_return_t sfi_defer(uint64_t);

extern int sfi_ledger_entry_add(ledger_template_t template, sfi_class_id_t class_id);
#endif /* MACH_KERNEL_PRIVATE */
#endif /* XNU_KERNEL_PRIVATE */

#endif /* _KERN_SFI_H_ */
