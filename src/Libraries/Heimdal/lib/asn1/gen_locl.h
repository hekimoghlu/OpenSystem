/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 28, 2022.
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

#ifndef __GEN_LOCL_H__
#define __GEN_LOCL_H__

#include <config.h>

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <ctype.h>
#include <time.h>
#include <errno.h>
#include <err.h>
#include <roken.h>
#include "hash.h"
#include "symbol.h"
#include "asn1-common.h"
#include "der.h"
#include "der-private.h"

void generate_type (const Symbol *);
void generate_constant (const Symbol *);
void generate_type_encode (const Symbol *);
void generate_type_decode (const Symbol *);
void generate_type_free (const Symbol *);
void generate_type_length (const Symbol *);
void generate_type_copy (const Symbol *);
void generate_type_seq (const Symbol *);
void generate_glue (const Type *, const char*);

void check_preserve_type(const char *, Type *);

const char *classname(Der_class);
const char *valuename(Der_class, int);

void gen_compare_defval(const char *, struct value *);
void gen_assign_defval(const char *, struct value *);


void init_generate (const char *, const char *);
const char *get_filename (void);
void close_generate(void);
void add_import(const char *);
void add_export(const char *);
int is_export(const char *);
int yyparse(void);
int is_primitive_type(int);

int preserve_type(const char *);
int seq_type(const char *);
int extra_data_type(const char *p);

void generate_header_of_codefile(const char *);
void close_codefile(void);

int is_template_compat (const Symbol *);
void generate_template(const Symbol *);
void gen_template_import(const Symbol *);


extern FILE *privheaderfile, *headerfile, *codefile, *logfile, *templatefile;
extern const char *fuzzer_string;
extern int support_ber;
extern int template_flag;
extern int rfc1510_bitstring;
extern int one_code_file;
extern int foundation_flag;
extern int parse_units_flag;
extern char *type_file_string;

extern int error_flag;

#endif /* __GEN_LOCL_H__ */
