/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 13, 2022.
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
#line 23 "eval.def"

#line 29 "eval.def"

#include <config.h>
#if defined (HAVE_UNISTD_H)
#  ifdef _MINIX
#    include <sys/types.h>
#  endif
#  include <unistd.h>
#endif

#include "../shell.h"
#include "bashgetopt.h"
#include "common.h"

/* Parse the string that these words make, and execute the command found. */
int
eval_builtin (list)
     WORD_LIST *list;
{
  if (no_options (list))
    return (EX_USAGE);
  list = loptend;	/* skip over possible `--' */

  /* Note that parse_and_execute () frees the string it is passed. */
  return (list ? parse_and_execute (string_list (list), "eval", SEVAL_NOHIST) : EXECUTION_SUCCESS);
}
