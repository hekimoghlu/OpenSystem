/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 22, 2024.
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

#if 0 /*===-- include/flang/Runtime/magic-numbers.h -----------------------===*/
/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://toolchain.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===----------------------------------------------------------------------===*/
#endif
#if 0
This header can be included into both Fortran and C.

This file defines various code values that need to be exported
to predefined Fortran standard modules as well as to C/C++
code in the compiler and runtime library.
These include:
 - the error/end code values that can be returned
   to an IOSTAT= or STAT= specifier on a Fortran I/O statement
   or coindexed data reference (see Fortran 2018 12.11.5,
   16.10.2, and 16.10.2.33)
Codes from <errno.h>, e.g. ENOENT, are assumed to be positive
and are used "raw" as IOSTAT values.

CFI_ERROR_xxx and CFI_INVALID_xxx macros from ISO_Fortran_binding.h
have small positive values.  The FORTRAN_RUNTIME_STAT_xxx macros here
start at 100 so as to never conflict with those codes.
#endif
#ifndef LANGUAGE_COMPABILITY_RUNTIME_MAGIC_NUMBERS_H_
#define LANGUAGE_COMPABILITY_RUNTIME_MAGIC_NUMBERS_H_

#define LANGUAGE_COMPABILITY_DEFAULT_OUTPUT_UNIT 6
#define LANGUAGE_COMPABILITY_DEFAULT_INPUT_UNIT 5
#define LANGUAGE_COMPABILITY_ERROR_UNIT 0

#define LANGUAGE_COMPABILITY_RUNTIME_IOSTAT_END (-1)
#define LANGUAGE_COMPABILITY_RUNTIME_IOSTAT_EOR (-2)
#define LANGUAGE_COMPABILITY_RUNTIME_IOSTAT_FLUSH (-3)
#define LANGUAGE_COMPABILITY_RUNTIME_IOSTAT_INQUIRE_INTERNAL_UNIT 256

#define LANGUAGE_COMPABILITY_RUNTIME_STAT_FAILED_IMAGE 101
#define LANGUAGE_COMPABILITY_RUNTIME_STAT_LOCKED 102
#define LANGUAGE_COMPABILITY_RUNTIME_STAT_LOCKED_OTHER_IMAGE 103
#define LANGUAGE_COMPABILITY_RUNTIME_STAT_STOPPED_IMAGE 104
#define LANGUAGE_COMPABILITY_RUNTIME_STAT_UNLOCKED 105
#define LANGUAGE_COMPABILITY_RUNTIME_STAT_UNLOCKED_FAILED_IMAGE 106

#if 0
Status codes for GET_COMMAND_ARGUMENT. The status for 'value too short' needs
to be -1, the others must be positive.
#endif
#define LANGUAGE_COMPABILITY_RUNTIME_STAT_INVALID_ARG_NUMBER 107
#define LANGUAGE_COMPABILITY_RUNTIME_STAT_MISSING_ARG 108
#define LANGUAGE_COMPABILITY_RUNTIME_STAT_VALUE_TOO_SHORT -1

#if 0
Status codes for GET_ENVIRONMENT_VARIABLE. Values mandated by the standard.
#endif
#define LANGUAGE_COMPABILITY_RUNTIME_STAT_MISSING_ENV_VAR 1
#define LANGUAGE_COMPABILITY_RUNTIME_STAT_ENV_VARS_UNSUPPORTED 2

#if 0
Processor-defined status code for MOVE_ALLOC where arguments are the
same allocatable.
#endif
#define LANGUAGE_COMPABILITY_RUNTIME_STAT_MOVE_ALLOC_SAME_ALLOCATABLE 109

#if 0
Additional status code for a bad pointer DEALLOCATE.
#endif
#define LANGUAGE_COMPABILITY_RUNTIME_STAT_BAD_POINTER_DEALLOCATION 110

#if 0
Status codes for GETCWD.
#endif
#define LANGUAGE_COMPABILITY_RUNTIME_STAT_MISSING_CWD 111

#if 0
ieee_class_type values
The sequence is that of F18 Clause 17.2p3, but nothing depends on that.
#endif
#define _FORTRAN_RUNTIME_IEEE_SIGNALING_NAN 1
#define _FORTRAN_RUNTIME_IEEE_QUIET_NAN 2
#define _FORTRAN_RUNTIME_IEEE_NEGATIVE_INF 3
#define _FORTRAN_RUNTIME_IEEE_NEGATIVE_NORMAL 4
#define _FORTRAN_RUNTIME_IEEE_NEGATIVE_SUBNORMAL 5
#define _FORTRAN_RUNTIME_IEEE_NEGATIVE_ZERO 6
#define _FORTRAN_RUNTIME_IEEE_POSITIVE_ZERO 7
#define _FORTRAN_RUNTIME_IEEE_POSITIVE_SUBNORMAL 8
#define _FORTRAN_RUNTIME_IEEE_POSITIVE_NORMAL 9
#define _FORTRAN_RUNTIME_IEEE_POSITIVE_INF 10
#define _FORTRAN_RUNTIME_IEEE_OTHER_VALUE 11

#if 0
ieee_flag_type values
The values are those of a common but not universal fenv.h file.
The denorm value is a nonstandard extension.
#endif
#define _FORTRAN_RUNTIME_IEEE_INVALID 1
#define _FORTRAN_RUNTIME_IEEE_DENORM 2
#define _FORTRAN_RUNTIME_IEEE_DIVIDE_BY_ZERO 4
#define _FORTRAN_RUNTIME_IEEE_OVERFLOW 8
#define _FORTRAN_RUNTIME_IEEE_UNDERFLOW 16
#define _FORTRAN_RUNTIME_IEEE_INEXACT 32
#define _FORTRAN_RUNTIME_IEEE_ALL \
  _FORTRAN_RUNTIME_IEEE_INVALID | _FORTRAN_RUNTIME_IEEE_DENORM | \
      _FORTRAN_RUNTIME_IEEE_DIVIDE_BY_ZERO | _FORTRAN_RUNTIME_IEEE_OVERFLOW | \
      _FORTRAN_RUNTIME_IEEE_UNDERFLOW | _FORTRAN_RUNTIME_IEEE_INEXACT

#if 0
ieee_round_type values
The values are those of the toolchain.get.rounding intrinsic, which is assumed by
ieee_arithmetic module rounding procedures.
#endif
#define _FORTRAN_RUNTIME_IEEE_TO_ZERO 0
#define _FORTRAN_RUNTIME_IEEE_NEAREST 1
#define _FORTRAN_RUNTIME_IEEE_UP 2
#define _FORTRAN_RUNTIME_IEEE_DOWN 3
#define _FORTRAN_RUNTIME_IEEE_AWAY 4
#define _FORTRAN_RUNTIME_IEEE_OTHER 5

#if 0
INTEGER(kind=4) extents for ieee_exceptions module types ieee_modes_type and
ieee_status_type. These extent values are large enough to hold femode_t and
fenv_t data in many environments. An environment that does not meet these
size constraints may allocate memory with runtime size values.
#endif
#define _FORTRAN_RUNTIME_IEEE_FEMODE_T_EXTENT 2
#define _FORTRAN_RUNTIME_IEEE_FENV_T_EXTENT 8

#endif
