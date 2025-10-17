/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 27, 2022.
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
**
**  NAME
**
**      ERRORS.H
**
**  FACILITY:
**
**      Interface Definition Language (IDL) Compiler
**
**  ABSTRACT:
**
**
**
**  VERSION: DCE 1.0
**
*/

#ifndef ERRORS_H
#define ERRORS_H

#include <errno.h>
#include <nidl.h>
#include <nametbl.h>

#define IDL_ERROR_LIST_SIZE 5

/* An opaque pointer. */
#ifndef YY_TYPEDEF_YY_SCANNER_T
#define YY_TYPEDEF_YY_SCANNER_T
typedef void* yyscan_t;
#endif

void error
(
	long msg_id,
	...
);

void warning
(
	long msg_id,
	...
);

void vlog_source_error
(
 STRTAB_str_t filename,
 int lineno,
 long msg_id,
 va_list ap
);

void log_source_error
(
	/* it is not a nonsense */
	STRTAB_str_t filename,
	int lineno,
	long msg_id,
	... /* 0..5 args terminated by NULL if less than five */
);

void vlog_source_warning
(
 STRTAB_str_t filename,
 int lineno,
 long msg_id,
 va_list ap
 );

void log_source_warning
(
	/* it is not a nonsense */
	STRTAB_str_t filename,
	int lineno,
	long msg_id,
	... /* 0..5 args terminated by NULL if less than five */
);

void vlog_error
(
 /* it is not a nonsense */
 int lineno, /* Source line number */
 long msg_id, /* Message ID */
 va_list ap
);

void log_error
(
	/* it is not a nonsense */
	int lineno, /* Source line number */
	long msg_id, /* Message ID */
	... /* 0..5 args terminated by NULL if less than five */
);

void vlog_warning
(
 /* it is not a nonsense */
 int lineno, /* Source line number */
 long msg_id, /* Message ID */
 va_list ap
);

void log_warning
(
	/* it is not a nonsense */
	int lineno, /* Source line number */
	long msg_id, /* Message ID */
	... /* 0..5 args terminated by NULL if less than five */
);

typedef struct {
    long msg_id;
    const void* arg[IDL_ERROR_LIST_SIZE];
} idl_error_list_t;

typedef idl_error_list_t *idl_error_list_p;

void error_list
(
    int vecsize,
    idl_error_list_p errvec,
    boolean exitflag
);

void inq_name_for_errors
(
    char *name,
	size_t name_len
);

void set_name_for_errors
(
    char const *name
);

boolean print_errors
(
    void
);

#if ! defined YYLTYPE && ! defined YYLTYPE_IS_DECLARED
typedef struct YYLTYPE
{
    int first_line;
    int first_column;
    int last_line;
    int last_column;
} YYLTYPE;
# define yyltype YYLTYPE /* obsolescent; will be withdrawn */
# define YYLTYPE_IS_DECLARED 1
# define YYLTYPE_IS_TRIVIAL 1
#endif

struct parser_location_t;

void idl_yyerror
(
    const struct parser_location_t * location,
    char const * message
);

void yywhere
(
    const struct parser_location_t * location
);

/*
 * Error info to be fillin the fe_info nodes
 */
extern int          error_count;

/* XXX These globals are set and unset in the NIDL and ACF parsers in
 * xxx_input() and xxx_parser_destroy(). They should be more-or-less
 * accurate, but it would be better to continue plumbing the
 * parser_location_t down into error() and warning() so that we can
 * get rid of this ugliness.
 */

extern FILE    *yyin_p;           /* Points to yyin or acf_yyin */
extern unsigned*yylineno_p;       /* Points to yylineno or acf_yylineno */

#ifdef DUMPERS
#define INTERNAL_ERROR(string) {printf("Internal Error Diagnostic: %s\n",string);warning(NIDL_INTERNAL_ERROR,__FILE__,__LINE__);}
#else
#define INTERNAL_ERROR(string) {error(NIDL_INTERNAL_ERROR,__FILE__,__LINE__); printf(string);}
#endif
#endif
/* preserve coding style vim: set tw=78 sw=4 : */
