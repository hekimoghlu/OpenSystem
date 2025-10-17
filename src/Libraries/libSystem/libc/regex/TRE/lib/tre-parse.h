/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 18, 2022.
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
#ifndef TRE_PARSE_H
#define TRE_PARSE_H 1

#ifdef __LIBC__
#include <xlocale_private.h>
#else /* !__LIBC__ */
#include <xlocale.h>
#endif /* !__LIBC__ */

/* Parse context. */
typedef struct {
  /* Memory allocator.	The AST is allocated using this. */
  tre_mem_t mem;
  /* Stack used for keeping track of regexp syntax. */
  tre_stack_t *stack;
  /* The parse result. */
  tre_ast_node_t *result;
  /* The regexp to parse and its length. */
  const tre_char_t *re;
  /* The first character of the entire regexp. */
  const tre_char_t *re_start;
  /* The first character after the end of the regexp. */
  const tre_char_t *re_end;
  /* The current locale */
  locale_t loc;
  int len;
  /* Current submatch ID. */
  int submatch_id;
  /* Current invisible submatch ID. */
  int submatch_id_invisible;
  /* Current position (number of literal). */
  int position;
  /* The highest back reference or -1 if none seen so far. */
  int max_backref;
  /* Number of tags that need reordering. */
  int num_reorder_tags;
  /* This flag is set if the regexp uses approximate matching. */
  int have_approx;
  /* Compilation flags. */
  int cflags;
  /* If this flag is set the top-level submatch is not captured. */
  int nofirstsub;
  /* The currently set approximate matching parameters. */
  int params[TRE_PARAM_LAST];
} tre_parse_ctx_t;

/* Parses a wide character regexp pattern into a syntax tree.  This parser
   handles both syntaxes (BRE and ERE), including the TRE extensions. */
__private_extern__ reg_errcode_t
tre_parse(tre_parse_ctx_t *ctx);

#endif /* TRE_PARSE_H */

/* EOF */
