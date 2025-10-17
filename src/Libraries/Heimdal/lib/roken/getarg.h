/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 25, 2025.
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

#ifndef __GETARG_H__
#define __GETARG_H__

#include <stddef.h>

#ifndef ROKEN_LIB_FUNCTION
#ifdef _WIN32
#define ROKEN_LIB_FUNCTION
#define ROKEN_LIB_CALL     __cdecl
#else
#define ROKEN_LIB_FUNCTION
#define ROKEN_LIB_CALL
#endif
#endif

struct getargs{
    const char *long_name;
    char short_name;
    enum { arg_integer,
	   arg_string,
	   arg_flag,
	   arg_negative_flag,
	   arg_strings,
	   arg_double,
	   arg_collect,
	   arg_counter
    } type;
    void *value;
    const char *help;
    const char *arg_help;
};

enum {
    ARG_ERR_NO_MATCH  = 1,
    ARG_ERR_BAD_ARG,
    ARG_ERR_NO_ARG
};

typedef struct getarg_strings {
    int num_strings;
    char **strings;
} getarg_strings;

typedef int (*getarg_collect_func)(int short_opt,
				   int argc,
				   char **argv,
				   int *goptind,
				   int *goptarg,
				   void *data);

typedef struct getarg_collect_info {
    getarg_collect_func func;
    void *data;
} getarg_collect_info;

ROKEN_LIB_FUNCTION int ROKEN_LIB_CALL
getarg(struct getargs *args, size_t num_args,
       int argc, char **argv, int *goptind);

ROKEN_LIB_FUNCTION void ROKEN_LIB_CALL
arg_printusage (struct getargs *args,
		size_t num_args,
		const char *progname,
		const char *extra_string);

ROKEN_LIB_FUNCTION void ROKEN_LIB_CALL
arg_printusage_i18n (struct getargs *args,
		     size_t num_args,
		     const char *usage,
		     const char *progname,
		     const char *extra_string,
		     char *(*i18n)(const char *));

ROKEN_LIB_FUNCTION void ROKEN_LIB_CALL
free_getarg_strings (getarg_strings *);

#endif /* __GETARG_H__ */
