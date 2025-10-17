/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 24, 2022.
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
 * strtold() implementation
 */

#define S2F_function	strtold
#define S2F_type	2

/*
 * ast strtold() => strtod() when double == long double
 */

#define _AST_STD_H	1

#include <ast_common.h>

#if _ast_fltmax_double
#define strtold		______strtold
#endif

#include <ast_lib.h>
#include <ast_sys.h>

#if _ast_fltmax_double
#undef	strtold
#endif

#undef	_AST_STD_H

#include <ast.h>

#include "sfstrtof.h"
