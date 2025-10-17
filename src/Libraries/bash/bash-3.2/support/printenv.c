/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 7, 2025.
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
/* Copyright (C) 1997-2002 Free Software Foundation, Inc.

   This file is part of GNU Bash, the Bourne Again SHell.

   Bash is free software; you can redistribute it and/or modify it under
   the terms of the GNU General Public License as published by the Free
   Software Foundation; either version 2, or (at your option) any later
   version.

   Bash is distributed in the hope that it will be useful, but WITHOUT ANY
   WARRANTY; without even the implied warranty of MERCHANTABILITY or
   FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
   for more details.

   You should have received a copy of the GNU General Public License along
   with Bash; see the file COPYING.  If not, write to the Free Software
   Foundation, 59 Temple Place, Suite 330, Boston, MA 02111 USA. */

#if defined (HAVE_CONFIG_H)
#  include  <config.h>
#endif

#include "bashansi.h"

extern char **environ;

int
main (argc, argv) 
     int argc;
     char **argv;
{
  register char **envp, *eval;
  int len;

  argv++;
  argc--;

  /* printenv */
  if (argc == 0)
    {
      for (envp = environ; *envp; envp++)
	puts (*envp);
      exit (0);
    }

  /* printenv varname */
  len = strlen (*argv);
  for (envp = environ; *envp; envp++)
    {
      if (**argv == **envp && strncmp (*envp, *argv, len) == 0)
	{
	  eval = *envp + len;
	  /* If the environment variable doesn't have an `=', ignore it. */
	  if (*eval == '=')
	    {
	      puts (eval + 1);
	      exit (0);
	    }
	}
    }
  exit (1);
}
  

