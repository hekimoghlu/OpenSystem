/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 22, 2023.
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
/* $Id: cparse.h 11097 2009-01-30 10:27:37Z bhy $ */

#ifndef SWIG_CPARSE_H_
#define SWIG_CPARSE_H_

#include "swig.h"
#include "swigwarn.h"

#ifdef __cplusplus
extern "C" {
#endif

/* cscanner.c */
  extern String *cparse_file;
  extern int cparse_line;
  extern int cparse_cplusplus;
  extern int cparse_start_line;

  extern void Swig_cparse_cplusplus(int);
  extern void scanner_file(File *);
  extern void scanner_next_token(int);
  extern void skip_balanced(int startchar, int endchar);
  extern void skip_decl(void);
  extern void scanner_check_typedef(void);
  extern void scanner_ignore_typedef(void);
  extern void scanner_last_id(int);
  extern void scanner_clear_rename(void);
  extern void scanner_set_location(String *file, int line);
  extern void scanner_set_main_input_file(String *file);
  extern String *scanner_get_main_input_file(void);
  extern void Swig_cparse_follow_locators(int);
  extern void start_inline(char *, int);
  extern String *scanner_ccode;
  extern int yylex(void);

/* parser.y */
  extern SwigType *Swig_cparse_type(String *);
  extern Node *Swig_cparse(File *);
  extern Hash *Swig_cparse_features(void);
  extern void SWIG_cparse_set_compact_default_args(int defargs);
  extern int SWIG_cparse_template_reduce(int treduce);

/* util.c */
  extern void Swig_cparse_replace_descriptor(String *s);
  extern void cparse_normalize_void(Node *);
  extern Parm *Swig_cparse_parm(String *s);
  extern ParmList *Swig_cparse_parms(String *s);


/* templ.c */
  extern int Swig_cparse_template_expand(Node *n, String *rname, ParmList *tparms, Symtab *tscope);
  extern Node *Swig_cparse_template_locate(String *name, ParmList *tparms, Symtab *tscope);
  extern void Swig_cparse_debug_templates(int);

#ifdef __cplusplus
}
#endif
#define SWIG_WARN_NODE_BEGIN(Node) \
 { \
  String *wrnfilter = Node ? Getattr(Node,"feature:warnfilter") : 0; \
  if (wrnfilter) Swig_warnfilter(wrnfilter,1)
#define SWIG_WARN_NODE_END(Node) \
  if (wrnfilter) Swig_warnfilter(wrnfilter,0); \
 }
#endif
