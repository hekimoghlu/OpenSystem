/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 2, 2024.
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
/* $Id$ */

#ifndef __PARSE_TIME_H__
#define __PARSE_TIME_H__

#ifndef ROKEN_LIB_FUNCTION
#ifdef _WIN32
#define ROKEN_LIB_FUNCTION
#define ROKEN_LIB_CALL     __cdecl
#else
#define ROKEN_LIB_FUNCTION
#define ROKEN_LIB_CALL
#endif
#endif

ROKEN_LIB_FUNCTION int ROKEN_LIB_CALL
parse_time (const char *s, const char *def_unit);

ROKEN_LIB_FUNCTION size_t ROKEN_LIB_CALL
unparse_time (int t, char *s, size_t len);

ROKEN_LIB_FUNCTION size_t ROKEN_LIB_CALL
unparse_time_approx (int t, char *s, size_t len);

ROKEN_LIB_FUNCTION void ROKEN_LIB_CALL
print_time_table (FILE *f);

#endif /* __PARSE_TIME_H__ */
