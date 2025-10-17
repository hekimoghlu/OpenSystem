/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 14, 2025.
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
#line 23 "break.def"

#line 30 "break.def"
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

extern char *this_command_name;
extern int posixly_correct;

static int check_loop_level __P((void));

/* The depth of while's and until's. */
int loop_level = 0;

/* Non-zero when a "break" instruction is encountered. */
int breaking = 0;

/* Non-zero when we have encountered a continue instruction. */
int continuing = 0;

/* Set up to break x levels, where x defaults to 1, but can be specified
   as the first argument. */
int
break_builtin (list)
     WORD_LIST *list;
{
  intmax_t newbreak;

  if (check_loop_level () == 0)
    return (EXECUTION_SUCCESS);

  newbreak = get_numeric_arg (list, 1);

  if (newbreak <= 0)
    {
      sh_erange (list->word->word, "loop count");
      breaking = loop_level;
      return (EXECUTION_FAILURE);
    }

  if (newbreak > loop_level)
    newbreak = loop_level;

  breaking = newbreak;

  return (EXECUTION_SUCCESS);
}

#line 92 "break.def"

/* Set up to continue x levels, where x defaults to 1, but can be specified
   as the first argument. */
int
continue_builtin (list)
     WORD_LIST *list;
{
  intmax_t newcont;

  if (check_loop_level () == 0)
    return (EXECUTION_SUCCESS);

  newcont = get_numeric_arg (list, 1);

  if (newcont <= 0)
    {
      sh_erange (list->word->word, "loop count");
      breaking = loop_level;
      return (EXECUTION_FAILURE);
    }

  if (newcont > loop_level)
    newcont = loop_level;

  continuing = newcont;

  return (EXECUTION_SUCCESS);
}

/* Return non-zero if a break or continue command would be okay.
   Print an error message if break or continue is meaningless here. */
static int
check_loop_level ()
{
#if defined (BREAK_COMPLAINS)
  if (loop_level == 0 && posixly_correct == 0)
    builtin_error ("%s", _("only meaningful in a `for', `while', or `until' loop"));
#endif /* BREAK_COMPLAINS */

  return (loop_level);
}
