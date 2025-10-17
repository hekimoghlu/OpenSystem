/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 20, 2023.
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
#line 23 "test.def"

#line 93 "test.def"

#line 101 "test.def"

#include <config.h>

#if defined (HAVE_UNISTD_H)
#  ifdef _MINIX
#    include <sys/types.h>
#  endif
#  include <unistd.h>
#endif

#include "../bashansi.h"

#include "../shell.h"
#include "../test.h"
#include "common.h"

extern char *this_command_name;

/* TEST/[ builtin. */
int
test_builtin (list)
     WORD_LIST *list;
{
  char **argv;
  int argc, result;

  /* We let Matthew Bradburn and Kevin Braunsdorf's code do the
     actual test command.  So turn the list of args into an array
     of strings, since that is what their code wants. */
  if (list == 0)
    {
      if (this_command_name[0] == '[' && !this_command_name[1])
	{
	  builtin_error ("missing `]'");
	  return (EX_BADUSAGE);
	}

      return (EXECUTION_FAILURE);
    }

  argv = make_builtin_argv  (list, &argc);
  result = test_command (argc, argv);
  free ((char *)argv);

  return (result);
}
