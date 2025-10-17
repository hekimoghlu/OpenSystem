/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 6, 2023.
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
#pragma once

#include <assert.h>
#include <string.h>

#if __cplusplus
extern "C" {
#endif

#define COROUTINE __attribute__((noreturn)) void

const size_t COROUTINE_REGISTERS = 9;

typedef struct
{
    void **stack_pointer;
} coroutine_context;

typedef COROUTINE(* coroutine_start)(coroutine_context *from, coroutine_context *self);

static inline void coroutine_initialize(
    coroutine_context *context,
    coroutine_start start,
    void *stack_pointer,
    size_t stack_size
) {
    context->stack_pointer = (void**)stack_pointer;

    if (!start) {
        assert(!context->stack_pointer);
        /* We are main coroutine for this thread */
        return;
    }

    *--context->stack_pointer = (void*)start;

    context->stack_pointer -= COROUTINE_REGISTERS;
    memset(context->stack_pointer, 0, sizeof(void*) * COROUTINE_REGISTERS);
}

coroutine_context * coroutine_transfer(coroutine_context * current, coroutine_context * target);

static inline void coroutine_destroy(coroutine_context * context)
{
}

#if __cplusplus
}
#endif
