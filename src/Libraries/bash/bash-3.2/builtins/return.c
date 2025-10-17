/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 12, 2025.
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

extern int last_command_exit_value;
extern int subshell_environment;
extern int return_catch_flag, return_catch_value;

/* If we are executing a user-defined function then exit with the value
   specified as an argument.  if no argument is given, then the last
   exit status is used. */
int
return_builtin (list)
     WORD_LIST *list;
{
  return_catch_value = get_exitstat (list);

  if (return_catch_flag)
    longjmp (return_catch, 1);
  else
    {
      builtin_error ("%s", _("can only `return' from a function or sourced script"));
      return (EXECUTION_FAILURE);
    }
}
