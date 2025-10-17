/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 27, 2022.
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
#line 23 "shift.def"

#include <config.h>

#if defined (HAVE_UNISTD_H)
#  ifdef _MINIX
#    include <sys/types.h>
#  endif
#  include <unistd.h>
#endif

#include "../bashansi.h"
#include "../bashintl.h"

#include "../shell.h"
#include "common.h"

#line 45 "shift.def"

int print_shift_error;

/* Shift the arguments ``left''.  Shift DOLLAR_VARS down then take one
   off of REST_OF_ARGS and place it into DOLLAR_VARS[9].  If LIST has
   anything in it, it is a number which says where to start the
   shifting.  Return > 0 if `times' > $#, otherwise 0. */
int
shift_builtin (list)
     WORD_LIST *list;
{
  intmax_t times;
  register int count;
  WORD_LIST *temp;

  times = get_numeric_arg (list, 0);

  if (times == 0)
    return (EXECUTION_SUCCESS);
  else if (times < 0)
    {
      sh_erange (list ? list->word->word : NULL, _("shift count"));
      return (EXECUTION_FAILURE);
    }
  else if (times > number_of_args ())
    {
      if (print_shift_error)
	sh_erange (list ? list->word->word : NULL, _("shift count"));
      return (EXECUTION_FAILURE);
    }

  while (times-- > 0)
    {
      if (dollar_vars[1])
	free (dollar_vars[1]);

      for (count = 1; count < 9; count++)
	dollar_vars[count] = dollar_vars[count + 1];

      if (rest_of_args)
	{
	  temp = rest_of_args;
	  dollar_vars[9] = savestring (temp->word->word);
	  rest_of_args = rest_of_args->next;
	  temp->next = (WORD_LIST *)NULL;
	  dispose_words (temp);
	}
      else
	dollar_vars[9] = (char *)NULL;
    }
  return (EXECUTION_SUCCESS);
}
