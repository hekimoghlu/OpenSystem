/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 15, 2021.
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
#line 65 "let.def"

#include <config.h>

#if defined (HAVE_UNISTD_H)
#  ifdef _MINIX
#    include <sys/types.h>
#  endif
#  include <unistd.h>
#endif

#include "../bashintl.h"

#include "../shell.h"
#include "common.h"

/* Arithmetic LET function. */
int
let_builtin (list)
     WORD_LIST *list;
{
  intmax_t ret;
  int expok;

  /* Skip over leading `--' argument. */
  if (list && list->word && ISOPTION (list->word->word, '-'))
    list = list->next;

  if (list == 0)
    {
      builtin_error ("%s", _("expression expected"));
      return (EXECUTION_FAILURE);
    }

  for (; list; list = list->next)
    {
      ret = evalexp (list->word->word, &expok);
      if (expok == 0)
	return (EXECUTION_FAILURE);
    }

  return ((ret == 0) ? EXECUTION_FAILURE : EXECUTION_SUCCESS);
}

#ifdef INCLUDE_UNUSED
int
exp_builtin (list)
     WORD_LIST *list;
{
  char *exp;
  intmax_t ret;
  int expok;

  if (list == 0)
    {
      builtin_error ("%s", _("expression expected"));
      return (EXECUTION_FAILURE);
    }

  exp = string_list (list);
  ret = evalexp (exp, &expok);
  (void)free (exp);
  return (((ret == 0) || (expok == 0)) ? EXECUTION_FAILURE : EXECUTION_SUCCESS);
}
#endif
