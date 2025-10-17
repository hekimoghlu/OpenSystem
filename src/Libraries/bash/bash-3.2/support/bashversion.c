/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 9, 2023.
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
/* Copyright (C) 2001 Free Software Foundation, Inc.

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

#include "config.h"

#include "stdc.h"

#include <stdio.h>

#if defined (HAVE_UNISTD_H)
#  include <unistd.h>
#endif

#include "bashansi.h"

#include "version.h"
#include "conftypes.h"

#define RFLAG	0x0001
#define VFLAG	0x0002
#define MFLAG	0x0004
#define PFLAG	0x0008
#define SFLAG	0x0010
#define LFLAG	0x0020
#define XFLAG	0x0040

extern int optind;
extern char *optarg;

extern char *dist_version;
extern int patch_level;

char *shell_name = "bash";
char *progname;

static void
usage()
{
  fprintf(stderr, "%s: usage: %s [-hrvpmlsx]\n", progname, progname);
}

int
main (argc, argv)
     int argc;
     char **argv;
{
  int opt, oflags;
  char dv[128], *rv;

  if (progname = strrchr (argv[0], '/'))
    progname++;
  else
    progname = argv[0];

  oflags = 0;
  while ((opt = getopt(argc, argv, "hrvmpslx")) != EOF)
    {
      switch (opt)
	{
	case 'h':
	  usage ();
	  exit (0);
	case 'r':
	  oflags |= RFLAG;	/* release */
	  break;
	case 'v':
	  oflags |= VFLAG;	/* version */
	  break;
	case 'm':
	  oflags |= MFLAG;	/* machtype */
	  break;
	case 'p':
	  oflags |= PFLAG;	/* patchlevel */
	  break;
	case 's':		/* short version string */
	  oflags |= SFLAG;
	  break;
	case 'l':		/* long version string */
	  oflags |= LFLAG;
	  break;
	case 'x':		/* extended version information */
	  oflags |= XFLAG;
	  break;
	default:
	  usage ();
	  exit (2);
	}
    }

  argc -= optind;
  argv += optind;

  if (argc > 0)
    {
      usage ();
      exit (2);
    }

  /* default behavior */  
  if (oflags == 0)
    oflags = SFLAG;

  if (oflags & (RFLAG|VFLAG))
    {
      strcpy (dv, dist_version);
      rv = strchr (dv, '.');
      if (rv)
        *rv++ = '\0';
      else
        rv = "00";
    }
  if (oflags & RFLAG)
    printf ("%s\n", dv);
  else if (oflags & VFLAG)
    printf ("%s\n", rv);
  else if (oflags & MFLAG)
    printf ("%s\n", MACHTYPE);
  else if (oflags & PFLAG)
    printf ("%d\n", patch_level);
  else if (oflags & SFLAG)
    printf ("%s\n", shell_version_string ());
  else if (oflags & LFLAG)
    show_shell_version (0);
  else if (oflags & XFLAG)
    show_shell_version (1);

  exit (0);
}
