/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 16, 2025.
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
#include "uucp.h"

#if USE_RCS_ID
const char uuname_rcsid[] = "$Id: uuname.c,v 1.24 2002/03/05 19:10:42 ian Rel $";
#endif

#include "getopt.h"

#include "uudefs.h"
#include "uuconf.h"
#include "system.h"

/* Local functions.  */

static void unusage P((void));
static void unhelp P((void));

/* Long getopt options.  */
static const struct option asNlongopts[] =
{
  { "aliases", no_argument, NULL, 'a' },
  { "local", no_argument, NULL, 'l' },
  { "config", required_argument, NULL, 'I' },
  { "debug", required_argument, NULL, 'x' },
  { "version", no_argument, NULL, 'v' },
  { "help", no_argument, NULL, 1 },
  { NULL, 0, NULL, 0 }
};

int
main (argc, argv)
     int argc;
     char **argv;
{
  /* -a: display aliases.  */
  boolean falias = FALSE;
  /* -l: if true, output local node name.  */
  boolean flocal = FALSE;
  /* -I: configuration file name.  */
  const char *zconfig = NULL;
  int iopt;
  pointer puuconf;
  int iuuconf;

  if (argc < 1)
  {
      zProgram = "uuname";
      unusage ();
  }
  
  zProgram = argv[0];

  while ((iopt = getopt_long (argc, argv, "alI:vx:", asNlongopts,
			      (int *) NULL)) != EOF)
    {
      switch (iopt)
	{
	case 'a':
	  /* Display aliases.  */
	  falias = TRUE;
	  break;

	case 'l':
	  /* Output local node name.  */
	  flocal = TRUE;
	  break;

	case 'I':
	  /* Configuration file name.  */
	  if (fsysdep_other_config (optarg))
	    zconfig = optarg;
	  break;

	case 'x':
#if DEBUG > 1
	  /* Set debugging level.  */
	  iDebug |= idebug_parse (optarg);
#endif
	  break;

	case 'v':
	  /* Print version and exit.  */
	  printf ("uuname (Taylor UUCP) %s\n", VERSION);
	  printf ("Copyright (C) 1991, 92, 93, 94, 1995, 2002 Ian Lance Taylor\n");
	  printf ("This program is free software; you may redistribute it under the terms of\n");
	  printf ("the GNU General Public LIcense.  This program has ABSOLUTELY NO WARRANTY.\n");
	  exit (EXIT_SUCCESS);
	  /*NOTREACHED*/

	case 1:
	  /* --help.  */
	  unhelp ();
	  exit (EXIT_SUCCESS);
	  /*NOTREACHED*/

	case 0:
	  /* Long option found and flag set.  */
	  break;

	default:
	  unusage ();
	  /*NOTREACHED*/
	}
    }

  if (optind != argc)
    unusage ();

  iuuconf = uuconf_init (&puuconf, (const char *) NULL, zconfig);
  if (iuuconf != UUCONF_SUCCESS)
    ulog_uuconf (LOG_FATAL, puuconf, iuuconf);

#if DEBUG > 1
  {
    const char *zdebug;

    iuuconf = uuconf_debuglevel (puuconf, &zdebug);
    if (iuuconf != UUCONF_SUCCESS)
      ulog_uuconf (LOG_FATAL, puuconf, iuuconf);
    if (zdebug != NULL)
      iDebug |= idebug_parse (zdebug);
  }
#endif

  usysdep_initialize (puuconf, INIT_SUID | INIT_NOCHDIR);

  if (flocal)
    {
      const char *zlocalname;

      iuuconf = uuconf_localname (puuconf, &zlocalname);
      if (iuuconf == UUCONF_NOT_FOUND)
	{
	  zlocalname = zsysdep_localname ();
	  if (zlocalname == NULL)
	    usysdep_exit (FALSE);
	}
      else if (iuuconf != UUCONF_SUCCESS)
	ulog_uuconf (LOG_FATAL, puuconf, iuuconf);
      printf ("%s\n", zlocalname);
    }
  else
    {
      char **pznames, **pz;

      iuuconf = uuconf_system_names (puuconf, &pznames, falias);
      if (iuuconf != UUCONF_SUCCESS)
	ulog_uuconf (LOG_FATAL, puuconf, iuuconf);

      for (pz = pznames; *pz != NULL; pz++)
	printf ("%s\n", *pz);
    }

  usysdep_exit (TRUE);

  /* Avoid warnings about not returning a value.  */
  return 0;
}

/* Print a usage message and die.  */

static void
unusage ()
{
  fprintf (stderr,
	   "Usage: %s [-a] [-l] [-I file]\n", zProgram);
  fprintf (stderr, "Use %s --help for help\n", zProgram);
  exit (EXIT_FAILURE);
}

/* Print a help message.  */

static void unhelp ()
{
  printf ("Taylor UUCP %s, copyright (C) 1991, 92, 93, 94, 1995, 2002 Ian Lance Taylor\n",
	   VERSION);
  printf ("Usage: %s [-a] [-l] [-I file]\n", zProgram);
  printf (" -a,--aliases: display aliases\n");
  printf (" -l,--local: print local name\n");
#if HAVE_TAYLOR_CONFIG
  printf (" -I,--config file: Set configuration file to use\n");
#endif /* HAVE_TAYLOR_CONFIG */
  printf (" -v,--version: Print version and exit\n");
  printf (" --help: Print help and exit\n");
  printf ("Report bugs to taylor-uucp@gnu.org\n");
}
