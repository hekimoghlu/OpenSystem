/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 11, 2024.
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
#include <errno.h>

#include "builtins.h"
#include "shell.h"
#include "jobs.h"
#include "bashgetopt.h"

#ifndef errno
extern int errno;
#endif

extern int dollar_dollar_pid;
extern int last_command_exit_value;

int
push_builtin (list)
     WORD_LIST *list;
{
  pid_t pid;
  int xstatus, opt;

  xstatus = EXECUTION_SUCCESS;
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

  pid = make_child (savestring ("push"), 0);
  if (pid == -1)
    {
      builtin_error ("cannot fork: %s", strerror (errno));
      return (EXECUTION_FAILURE);
    }
  else if (pid == 0)
    {
      /* Shell variable adjustments: $SHLVL, $$, $PPID, $! */
      adjust_shell_level (1);
      dollar_dollar_pid = getpid ();
      set_ppid ();

      /* Clean up job control stuff. */
      stop_making_children ();
      cleanup_the_pipeline ();
      delete_all_jobs (0);

      last_asynchronous_pid = NO_PID;

      /* Make sure the job control code has the right values for
	 the shell's process group and tty process group, and that
	 the signals are set correctly for job control. */
      initialize_job_control (0);
      initialize_job_signals ();

      /* And read commands until exit. */
      reader_loop ();
      exit_shell (last_command_exit_value);
    }
  else
    {
      stop_pipeline (0, (COMMAND *)NULL);
      xstatus = wait_for (pid);
      return (xstatus);
    }   
}

char *push_doc[] = {
	"Create a child that is an exact duplicate of the running shell",
	"and wait for it to exit.  The $SHLVL, $!, $$, and $PPID variables",
	"are adjusted in the child.  The return value is the exit status",
	"of the child.",
	(char *)NULL
};

struct builtin push_struct = {
	"push",
	push_builtin,
	BUILTIN_ENABLED,
	push_doc,
	"push",
	0
};
