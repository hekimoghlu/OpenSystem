/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 25, 2021.
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
 * Copyright (c) 2001 by Sun Microsystems, Inc.
 * All rights reserved.
 */

#ifndef _STACK_H
#define	_STACK_H

/*
 * Routines for manipulating stacks
 */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct stk stk_t;

stk_t *stack_new(void (*)(void *));
void stack_free(stk_t *);
void *stack_pop(stk_t *);
void *stack_peek(stk_t *);
void stack_push(stk_t *, void *);
int stack_level(stk_t *);

#ifdef __cplusplus
}
#endif

#endif /* _STACK_H */
