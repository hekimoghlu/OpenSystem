/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 9, 2023.
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

#include <wtf/Assertions.h>

/* This file serves as the platform independent redirection header for
 * platform dependent register information. Each architecture has its own header.
 *
 * Each header defines a few important macros that are used in a platform independent
 * way - see for example jit/RegisterSet.cpp.
 * - FOR_EACH_GP_REGISTER which lists all available general purpose registers.
 * - FOR_EACH_FP_REGISTER which lists all available floating point registers.
 * these take themselves a macro that can filter through the available information
 * spread accross the four macro arguments.
 * = 1. id: is an identifier used to specify the register (for example, as an enumerator
 * in an enum);
 * = 2. name: is a string constant specifying the name of the identifier;
 * = 3. isReserved: a boolean (usually 0/1) specifying if this is a reserved register;
 * = 4. isCalleeSaved: a boolean (usually 0/1) specifying if this is a callee saved register;
 *
 * - A few other platform dependent macros can be specified to be used in platform
 *   dependent files (for example assembler X86Assembler.h).
 */

#if CPU(X86_64)
#include "X86_64Registers.h"
#elif CPU(ARM_THUMB2)
#include "ARMv7Registers.h"
#elif CPU(ARM64)
#include "ARM64Registers.h"
#else
    UNREACHABLE_FOR_PLATFORM();
#endif
