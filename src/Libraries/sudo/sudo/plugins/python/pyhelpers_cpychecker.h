/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 21, 2024.
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
#ifndef SUDO_PLUGIN_PYHELPERS_CPYCHECKER_H
#define	SUDO_PLUGIN_PYHELPERS_CPYCHECKER_H

/* Helper macros for cpychecker */

#if defined(WITH_CPYCHECKER_RETURNS_BORROWED_REF_ATTRIBUTE)
    #define CPYCHECKER_RETURNS_BORROWED_REF \
        __attribute__((cpychecker_returns_borrowed_ref))
#else
    #define CPYCHECKER_RETURNS_BORROWED_REF
#endif

#ifdef WITH_CPYCHECKER_NEGATIVE_RESULT_SETS_EXCEPTION_ATTRIBUTE
    #define CPYCHECKER_NEGATIVE_RESULT_SETS_EXCEPTION \
        __attribute__ ((cpychecker_negative_result_sets_exception))
#else
    #define CPYCHECKER_NEGATIVE_RESULT_SETS_EXCEPTION
#endif

#if defined(WITH_CPYCHECKER_STEALS_REFERENCE_TO_ARG_ATTRIBUTE)
  #define CPYCHECKER_STEALS_REFERENCE_TO_ARG(n) \
   __attribute__((cpychecker_steals_reference_to_arg(n)))
#else
 #define CPYCHECKER_STEALS_REFERENCE_TO_ARG(n)
#endif

#endif // SUDO_PLUGIN_PYHELPERS_CPYCHECKER_H
