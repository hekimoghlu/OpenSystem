/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 25, 2024.
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

#include "Register.h"
#include "StackAlignment.h"
#include <wtf/Assertions.h>

namespace JSC {

// The maxFrameExtentForSlowPathCall is the max amount of stack space (in bytes)
// that can be used for outgoing args when calling a slow path C function
// from JS code.

#if !ENABLE(ASSEMBLER)
static constexpr size_t maxFrameExtentForSlowPathCall = 0;

#elif CPU(X86_64)
// All args in registers. Windows also uses System V ABI.
static constexpr size_t maxFrameExtentForSlowPathCall = 0;

#elif CPU(ARM64) || CPU(ARM64E) || CPU(RISCV64)
// All args in registers.
static constexpr size_t maxFrameExtentForSlowPathCall = 0;

#elif CPU(ARM)
// First four args in registers, remaining 4 args on stack.
static constexpr size_t maxFrameExtentForSlowPathCall = 24;

#else
#error "Unsupported CPU: need value for maxFrameExtentForSlowPathCall"

#endif

static_assert(!(maxFrameExtentForSlowPathCall % sizeof(Register)), "Extent must be in multiples of registers");

#if ENABLE(ASSEMBLER)
// Make sure that cfr - maxFrameExtentForSlowPathCall bytes will make the stack pointer aligned
static_assert((maxFrameExtentForSlowPathCall % 16) == 16 - sizeof(CallerFrameAndPC), "Extent must align stack from callframe pointer");
#endif

static constexpr size_t maxFrameExtentForSlowPathCallInRegisters = maxFrameExtentForSlowPathCall / sizeof(Register);

} // namespace JSC
