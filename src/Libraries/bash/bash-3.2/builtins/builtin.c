/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 17, 2021.
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
#line 23 "builtin.def"

#line 31 "builtin.def"
#include <config.h>

#if defined (HAVE_UNISTD_H)
#  ifdef _MINIX
#    include <sys/types.h>
#  endif
#  include <unistd.h>
#endif

#include "../shell.h"
#include "common.h"
#include "bashgetopt.h"

extern char *this_command_name;

/* Run the command mentioned in list directly, without going through the
   normal alias/function/builtin/filename lookup process. */
int
builtin_builtin (list)
     WORD_LIST *list;
{
  sh_builtin_func_t *function;
  register char *command;

  if (no_options (list))
    return (EX_USAGE);
  list = loptend;	/* skip over possible `--' */

  if (list == 0)
    return (EXECUTION_SUCCESS);

  command = list->word->word;
#if defined (DISABLED_BUILTINS)
  function = builtin_address (command);
#else /* !DISABLED_BUILTINS */
  function = find_shell_builtin (command);
#endif /* !DISABLED_BUILTINS */

  if (!function)
    {
      sh_notbuiltin (command);
      return (EXECUTION_FAILURE);
    }
  else
    {
      this_command_name = command;
      list = list->next;
      return ((*function) (list));
    }
}
