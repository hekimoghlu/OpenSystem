/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 5, 2024.
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
#ifndef _AST_COMMON_H
#define _AST_COMMON_H	1

#include <ast_sa.h>
#include <sys/types.h>

#define Void_t	void
#define _ARG_(x)	x
#define _BEGIN_EXTERNS_
#define _END_EXTERNS_
#define __STD_C		1

#if _hdr_stdint
#include <stdint.h>
#else
#include <inttypes.h>
#endif

#if _hdr_unistd
#include <unistd.h>
#endif

#define _typ_int32_t	1
#ifdef _ast_int8_t
#define _typ_int64_t	1
#endif

#endif
