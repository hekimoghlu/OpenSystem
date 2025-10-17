/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 27, 2022.
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
#ifndef BC_ARGS_H
#define BC_ARGS_H

#include <status.h>
#include <opt.h>
#include <vm.h>

/**
 * Processes command-line arguments.
 * @param argc        How many arguments there are.
 * @param argv        The array of arguments.
 * @param exit_exprs  True if bc/dc should exit when there are expressions,
 *                    false otherwise.
 * @param scale       A pointer to return the scale that the arguments set, if
 *                    any.
 * @param ibase       A pointer to return the ibase that the arguments set, if
 *                    any.
 * @param obase       A pointer to return the obase that the arguments set, if
 *                    any.
 */
void
bc_args(int argc, char* argv[], bool exit_exprs, BcBigDig* scale,
        BcBigDig* ibase, BcBigDig* obase);

#if BC_ENABLED

#if DC_ENABLED

/// Returns true if the banner should be quieted.
#define BC_ARGS_SHOULD_BE_QUIET (BC_IS_DC || vm->exprs.len > 1)

#else // DC_ENABLED

/// Returns true if the banner should be quieted.
#define BC_ARGS_SHOULD_BE_QUIET (vm->exprs.len > 1)

#endif // DC_ENABLED

#else // BC_ENABLED

/// Returns true if the banner should be quieted.
#define BC_ARGS_SHOULD_BE_QUIET (BC_IS_DC)

#endif // BC_ENABLED

// A reference to the list of long options.
extern const BcOptLong bc_args_lopt[];

#endif // BC_ARGS_H
