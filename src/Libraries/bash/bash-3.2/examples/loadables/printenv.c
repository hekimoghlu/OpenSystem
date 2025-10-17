/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 9, 2025.
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

extern char **export_env;

int
printenv_builtin (list) 
     WORD_LIST *list;
{
  register char **envp;
  int opt;
  SHELL_VAR *var;

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

  /* printenv */
  if (list == 0)
    {
      maybe_make_export_env ();		/* this allows minimal code */
      for (envp = export_env; *envp; envp++)
	printf ("%s\n", *envp);
      return (EXECUTION_SUCCESS);
    }

  /* printenv varname */
  var = find_variable (list->word->word);
  if (var == 0 || (exported_p (var) == 0))
    return (EXECUTION_FAILURE);

  if (function_p (var))
    print_var_function (var);
  else
    print_var_value (var, 0);

  return (EXECUTION_SUCCESS);
}

char *printenv_doc[] = {
	"print values of environment variables",
	(char *)NULL
};

struct builtin printenv_struct = {
	"printenv",
	printenv_builtin,
	BUILTIN_ENABLED,
	printenv_doc,
	"printenv [varname]",
	0
};
