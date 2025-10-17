/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 19, 2024.
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
#line 23 "times.def"

#line 30 "times.def"

#include <config.h>

#if defined (HAVE_UNISTD_H)
#  ifdef _MINIX
#    include <sys/types.h>
#  endif
#  include <unistd.h>
#endif

#include <stdio.h>
#include "../bashtypes.h"
#include "../shell.h"

#include <posixtime.h>

#if defined (HAVE_SYS_TIMES_H)
#  include <sys/times.h>
#endif /* HAVE_SYS_TIMES_H */

#if defined (HAVE_SYS_RESOURCE_H) && !defined (RLIMTYPE)
#  include <sys/resource.h>
#endif

#include "common.h"

/* Print the totals for system and user time used. */
int
times_builtin (list)
     WORD_LIST *list;
{
#if defined (HAVE_GETRUSAGE) && defined (HAVE_TIMEVAL) && defined (RUSAGE_SELF)
  struct rusage self, kids;

  USE_VAR(list);

  if (no_options (list))
    return (EX_USAGE);

  getrusage (RUSAGE_SELF, &self);
  getrusage (RUSAGE_CHILDREN, &kids);	/* terminated child processes */

  print_timeval (stdout, &self.ru_utime);
  putchar (' ');
  print_timeval (stdout, &self.ru_stime);
  putchar ('\n');
  print_timeval (stdout, &kids.ru_utime);
  putchar (' ');
  print_timeval (stdout, &kids.ru_stime);
  putchar ('\n');

#else
#  if defined (HAVE_TIMES)
  /* This uses the POSIX.1/XPG5 times(2) interface, which fills in a 
     `struct tms' with values of type clock_t. */
  struct tms t;

  USE_VAR(list);

  if (no_options (list))
    return (EX_USAGE);

  times (&t);

  print_clock_t (stdout, t.tms_utime);
  putchar (' ');
  print_clock_t (stdout, t.tms_stime);
  putchar ('\n');
  print_clock_t (stdout, t.tms_cutime);
  putchar (' ');
  print_clock_t (stdout, t.tms_cstime);
  putchar ('\n');

#  else /* !HAVE_TIMES */

  USE_VAR(list);

  if (no_options (list))
    return (EX_USAGE);
  printf ("0.00 0.00\n0.00 0.00\n");

#  endif /* HAVE_TIMES */
#endif /* !HAVE_TIMES */

  return (EXECUTION_SUCCESS);
}
