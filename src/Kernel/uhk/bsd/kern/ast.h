/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 25, 2024.
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
/* CMU_HIST */

/*
 *      kern/ast.h: Definitions for Asynchronous System Traps.
 */

#ifndef _KERN_AST_H_
#define _KERN_AST_H_

#include <kern/thread.h>

struct task;

extern void act_set_astbsd(thread_t);
extern void bsd_ast(thread_t);

#define AST_KEVENT_RETURN_TO_KERNEL  0x0001
#define AST_KEVENT_REDRIVE_THREADREQ 0x0002
#define AST_KEVENT_WORKQ_QUANTUM_EXPIRED 0x0004

extern void kevent_ast(thread_t thread, uint16_t bits);
extern void act_set_astkevent(thread_t thread, uint16_t bits);
extern uint16_t act_clear_astkevent(thread_t thread, uint16_t bits);
extern bool act_set_ast_reset_pcs(struct task *task, thread_t thread);

#if CONFIG_DTRACE
extern void ast_dtrace_on(void);
#endif

extern void act_set_astproc_resource(thread_t);
extern void proc_filedesc_ast(task_t task);
#endif  /* _KERN_AST_H_ */
