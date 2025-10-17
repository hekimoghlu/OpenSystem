/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 23, 2022.
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
#ifndef _NET_SYSCTL_H_
 #define _NET_SYSCTL_H_

#include <sys/cdefs.h>
#include <sys/sysctl.h>
/*
 * Helper utilities for the sysctl procedures used
 * by bsd networking.
 */

/*
 * DECLARE_SYSCTL_HANDLER_ARG_ARRAY
 *
 * Helper macro to be invoked from a sysctl handler function.
 *
 * The macro compares the `arg2' sysctl handler argument
 * with the `expected_array_size' macro parameter.
 *
 * If `arg2' is equal to `expected_array_size', the macro will
 * define two local variables, the names of which are controlled
 * by macro parameters:
 *
 *    element_type  *`array_var';
 *    unsigned int   `len_var';
 *
 * The `array_var' local variable will be sized in accordance
 * to the parameters. It does not use `__sized_by` annotation,
 * to allow the body of the handler function to change the values,
 * if needed.
 *
 * If `arg2' is not equal to `expected_array_size', the macro
 * will return from the sysctl handler function with the
 * EINVAL code.
 */
#define DECLARE_SYSCTL_HANDLER_ARG_ARRAY(element_type,                     \
	    expected_array_size,                                               \
	    array_var,                                                         \
	    len_var)                                                           \
    unsigned int len_var = (unsigned int)arg2;                             \
    if (len_var != (expected_array_size)) {                                \
	    return EINVAL;                                                     \
    }                                                                      \
    element_type * array_var = __unsafe_forge_bidi_indexable(              \
	element_type *, arg1, len_var * sizeof(element_type))


 #endif /* _NET_SYSCTL_H_ */
