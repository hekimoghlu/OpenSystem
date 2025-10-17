/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 19, 2024.
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

#ifndef __COM_RIGHT_H__
#define __COM_RIGHT_H__

#ifndef KRB5_LIB
#ifndef KRB5_LIB_FUNCTION
#if defined(_WIN32)
#define KRB5_LIB_FUNCTION __declspec(dllimport)
#define KRB5_LIB_CALL __stdcall
#define KRB5_LIB_VARIABLE __declspec(dllimport)
#else
#define KRB5_LIB_FUNCTION
#define KRB5_LIB_CALL
#define KRB5_LIB_VARIABLE
#endif
#endif
#endif

#ifdef _WIN32
#define KRB5_CALLCONV __stdcall
#else
#define KRB5_CALLCONV
#endif

#ifdef __STDC__
#include <stdarg.h>
#endif

struct error_table {
    char const * const * msgs;
    long base;
    int n_msgs;
};

struct et_list {
    struct et_list *next;
    struct error_table *table;
};
extern struct et_list *_et_list;

KRB5_LIB_FUNCTION const char * KRB5_LIB_CALL
com_right (struct et_list *list, long code);

KRB5_LIB_FUNCTION const char * KRB5_LIB_CALL
com_right_r (struct et_list *list, long code, char *, size_t);

KRB5_LIB_FUNCTION void KRB5_LIB_CALL
initialize_error_table_r (struct et_list **, const char **, int, long);

KRB5_LIB_FUNCTION void KRB5_LIB_CALL
free_error_table (struct et_list *);

#endif /* __COM_RIGHT_H__ */
