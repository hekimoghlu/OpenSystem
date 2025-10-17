/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 25, 2024.
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
#include <stdio.h>

#include "builtins.h"
#include "shell.h"
#include "bashgetopt.h"

whoami_builtin (list)
     WORD_LIST *list;
{
  int opt;

  reset_internal_getopt ();
  while ((opt = internal_getopt (list, "")) != -1)
    {
      switch (opt)
	{
	default:
	  builtin_usage ();
	  return (EX_USAGE);
	}
    }
  list = loptend;
  if (list)
    {
      builtin_usage ();
      return (EX_USAGE);
    }

  if (current_user.user_name == 0)
    get_current_user_info ();
  printf ("%s\n", current_user.user_name);
  return (EXECUTION_SUCCESS);
}

char *whoami_doc[] = {
	"display name of current user",
	(char *)NULL
};

struct builtin whoami_struct = {
	"whoami",
	whoami_builtin,
	BUILTIN_ENABLED,
	whoami_doc,
	"whoami",
	0
};
