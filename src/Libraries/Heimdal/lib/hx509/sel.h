/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 17, 2024.
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
enum hx_expr_op {
    op_TRUE,
    op_FALSE,
    op_NOT,
    op_AND,
    op_OR,
    op_COMP,

    comp_EQ,
    comp_NE,
    comp_IN,
    comp_TAILEQ,

    expr_NUMBER,
    expr_STRING,
    expr_FUNCTION,
    expr_VAR,
    expr_WORDS
};

struct hx_expr {
    enum hx_expr_op	op;
    void		*arg1;
    void		*arg2;
};

struct hx_expr_input {
    const char *buf;
    size_t length;
    size_t offset;
    struct hx_expr *expr;
    char *error;
};

extern struct hx_expr_input _hx509_expr_input;

#define yyparse _hx509_sel_yyparse
#define yylex   _hx509_sel_yylex
#define yyerror _hx509_sel_yyerror
#define yylval  _hx509_sel_yylval
#define yychar  _hx509_sel_yychar
#define yydebug _hx509_sel_yydebug
#define yynerrs _hx509_sel_yynerrs
#define yywrap  _hx509_sel_yywrap

int  _hx509_sel_yyparse(void);
int  _hx509_sel_yylex(void);
void _hx509_sel_yyerror(const char *);

