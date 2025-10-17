/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 22, 2022.
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
/* Copyright (C) 1992-2006 Free Software Foundation, Inc.

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

#include <config.h>

#include <sys/types.h>
#include <signal.h>

#include <stdio.h>
#if defined (HAVE_STDLIB_H)
#  include <stdlib.h>
#else
#  include "ansi_stdlib.h"
#endif /* HAVE_STDLIB_H */

/* Duplicated from signames.c */
#if !defined (NSIG)
#  define NSIG 64
#endif

#define LASTSIG NSIG+2

/* Imported from signames.c */
extern void initialize_signames ();
extern char *signal_names[];

char *progname;

void
write_signames (stream)
     FILE *stream;
{
  register int i;

  fprintf (stream, "/* This file was automatically created by %s.\n",
	   progname);
  fprintf (stream, "   Do not edit.  Edit support/signames.c instead. */\n\n");
  fprintf (stream,
	   "/* A translation list so we can be polite to our users. */\n");
#if defined (CROSS_COMPILING)
  fprintf (stream, "extern char *signal_names[];\n\n");
  fprintf (stream, "extern void initialize_signames __P((void));\n\n");
#else
  fprintf (stream, "char *signal_names[NSIG + 4] = {\n");

  for (i = 0; i <= LASTSIG; i++)
    fprintf (stream, "    \"%s\",\n", signal_names[i]);

  fprintf (stream, "    (char *)0x0\n");
  fprintf (stream, "};\n\n");
  fprintf (stream, "#define initialize_signames()\n\n");
#endif
}

int
main (argc, argv)
     int argc;
     char **argv;
{
  char *stream_name;
  FILE *stream;

  progname = argv[0];

  if (argc == 1)
    {
      stream_name = "stdout";
      stream = stdout;
    }
  else if (argc == 2)
    {
      stream_name = argv[1];
      stream = fopen (stream_name, "w");
    }
  else
    {
      fprintf (stderr, "Usage: %s [output-file]\n", progname);
      exit (1);
    }

  if (!stream)
    {
      fprintf (stderr, "%s: %s: cannot open for writing\n",
	       progname, stream_name);
      exit (2);
    }

#if !defined (CROSS_COMPILING)
  initialize_signames ();
#endif
  write_signames (stream);
  exit (0);
}
