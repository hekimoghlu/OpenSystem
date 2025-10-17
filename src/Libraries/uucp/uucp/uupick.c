/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 24, 2025.
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
const char uupick_rcsid[] = "$Id: uupick.c,v 1.19 2002/03/05 19:10:42 ian Rel $";
#endif

#include <errno.h>

#include "getopt.h"

#include "uudefs.h"
#include "uuconf.h"
#include "system.h"

/* Local functions.  */

static void upmovedir P((const char *zfull, const char *zrelative,
			 pointer pinfo));
static void upmove P((const char *zfrom, const char *zto));

/* Long getopt options.  */
static const struct option asPlongopts[] =
{
  { "system", required_argument, NULL, 's' },
  { "config", required_argument, NULL, 'I' },
  { "debug", required_argument, NULL, 'x' },
  { "version", no_argument, NULL, 'v' },
  { "help", no_argument, NULL, 1 },
  { NULL, 0, NULL, 0 }
};

/* Local functions.  */

static void upusage P((void));
static void uphelp P((void));

int
main (argc, argv)
     int argc;
     char **argv;
{
  /* -s: system name.  */
  const char *zsystem = NULL;
  /* -I: configuration file name.  */
  const char *zconfig = NULL;
  int iopt;
  pointer puuconf;
  int iuuconf;
  const char *zpubdir;
  char *zfile, *zfrom, *zfull;
  char *zallsys;
  char ab[1000];
  boolean fquit;

  zProgram = "uupick";

  if (argc < 1)
      upusage ();

  while ((iopt = getopt_long (argc, argv, "I:s:vx:", asPlongopts,
			      (int *) NULL)) != EOF)
    {
      switch (iopt)
	{
	case 's':
	  /* System name to get files from.  */
	  zsystem = optarg;
	  break;

	case 'I':
	  /* Name configuration file.  */
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
	  printf ("uupick (Taylor UUCP) %s\n", VERSION);
	  printf ("Copyright (C) 1991, 92, 93, 94, 1995, 2002 Ian Lance Taylor\n");
	  printf ("This program is free software; you may redistribute it under the terms of\n");
	  printf ("the GNU General Public LIcense.  This program has ABSOLUTELY NO WARRANTY.\n");
	  exit (EXIT_SUCCESS);
	  /*NOTREACHED*/

	case 1:
	  /* --help.  */
	  uphelp ();
	  exit (EXIT_SUCCESS);
	  /*NOTREACHED*/

	case 0:
	  /* Long option found and flag set.  */
	  break;

	default:
	  upusage ();
	  /*NOTREACHED*/
	}
    }

  if (argc != optind)
    upusage ();

  iuuconf = uuconf_init (&puuconf, (const char *) NULL, zconfig);
  if (iuuconf != UUCONF_SUCCESS)
    ulog_uuconf (LOG_FATAL, puuconf, iuuconf);

  usysdep_initialize (puuconf, INIT_GETCWD | INIT_NOCHDIR);

  zpubdir = NULL;
  if (zsystem != NULL)
    {
      struct uuconf_system ssys;

      /* Get the public directory for the system.  If we can't find
         the system information, just use the standard public
         directory, since uupick is not setuid.  */
      iuuconf = uuconf_system_info (puuconf, zsystem, &ssys);
      if (iuuconf == UUCONF_SUCCESS)
	{
	  zpubdir = zbufcpy (ssys.uuconf_zpubdir);
	  (void) uuconf_system_free (puuconf, &ssys);
	}
    }
  if (zpubdir == NULL)
    {
      iuuconf = uuconf_pubdir (puuconf, &zpubdir);
      if (iuuconf != UUCONF_SUCCESS)
	ulog_uuconf (LOG_FATAL, puuconf, iuuconf);
    }

  if (! fsysdep_uupick_init (zsystem, zpubdir))
    usysdep_exit (FALSE);

  zallsys = NULL;
  fquit = FALSE;

  while (! fquit
	 && ((zfile = zsysdep_uupick (zsystem, zpubdir, &zfrom, &zfull))
	     != NULL))
    {
      boolean fdir;
      char *zto, *zlocal;
      FILE *e;
      boolean fcontinue;

      fdir = fsysdep_directory (zfull);

      do
	{
	  boolean fbadname;

	  fcontinue = FALSE;

	  if (zallsys == NULL
	      || strcmp (zallsys, zfrom) != 0)
	    {
	      if (zallsys != NULL)
		{
		  ubuffree (zallsys);
		  zallsys = NULL;
		}

	      printf ("from %s: %s %s ?\n", zfrom, fdir ? "dir" : "file",
		      zfile);

	      if (fgets (ab, sizeof ab, stdin) == NULL)
		break;
	    }

	  if (ab[0] == 'q')
	    {
	      fquit = TRUE;
	      break;
	    }

	  switch (ab[0])
	    {
	    case '\n':
	      break;

	    case 'd':
	      if (fdir)
		(void) fsysdep_rmdir (zfull);
	      else
		{
		  if (remove (zfull) != 0)
		    ulog (LOG_ERROR, "remove (%s): %s", zfull,
			  strerror(errno));
		}
	      break;

	    case 'm':
	    case 'a':
	      zto = ab + 1 + strspn (ab + 1, " \t");
	      zto[strcspn (zto, " \t\n")] = '\0';
	      zlocal = zsysdep_uupick_local_file (zto, &fbadname);
	      if (zlocal == NULL)
		{
		  if (! fbadname)
		    usysdep_exit (FALSE);
		  ulog (LOG_ERROR, "%s: bad local file name", zto);
		  fcontinue = TRUE;
		  break;
		}
	      zto = zsysdep_in_dir (zlocal, zfile);
	      ubuffree (zlocal);
	      if (zto == NULL)
		usysdep_exit (FALSE);
	      if (! fdir)
		upmove (zfull, zto);
	      else
		{
		  usysdep_walk_tree (zfull, upmovedir, (pointer) zto);
		  (void) fsysdep_rmdir (zfull);
		}
	      ubuffree (zto);

	      if (ab[0] == 'a')
		{
		  zallsys = zbufcpy (zfrom);
		  ab[0] = 'm';
		}

	      break;

	    case 'p':
	      if (fdir)
		ulog (LOG_ERROR, "Can't print directory");
	      else
		{
		  e = fopen (zfull, "r");
		  if (e == NULL)
		    ulog (LOG_ERROR, "fopen (%s): %s", zfull,
			  strerror (errno));
		  else
		    {
		      while (fgets (ab, sizeof ab, e) != NULL)
			(void) fputs (ab, stdout);
		      (void) fclose (e);
		    }
		}
	      fcontinue = TRUE;
	      break;

	    case '!':
	      (void) system (ab + 1);
	      fcontinue = TRUE;
	      break;

	    default:
	      printf ("uupick commands:\n");
	      printf ("q: quit\n");
	      printf ("<return>: skip file\n");
	      printf ("m [dir]: move file to directory\n");
	      printf ("a [dir]: move all files from this system to directory\n");
	      printf ("p: list file to stdout\n");
	      printf ("d: delete file\n");
	      printf ("! command: shell escape\n");
	      fcontinue = TRUE;
	      break;
	    }
	}
      while (fcontinue);

      ubuffree (zfull);
      ubuffree (zfrom);
      ubuffree (zfile);
    }

  (void) fsysdep_uupick_free (zsystem, zpubdir);

  usysdep_exit (TRUE);

  /* Avoid error about not returning.  */
  return 0;
}

/* Print usage message and die.  */

static void
upusage ()
{
  fprintf (stderr,
	   "Usage: %s [-s system] [-I config] [-x debug]\n", zProgram);
  fprintf (stderr, "Use %s --help for help\n", zProgram);
  exit (EXIT_FAILURE);
}

/* Print help message.  */

static void
uphelp ()
{
  printf ("Taylor UUCP %s, copyright (C) 1991, 92, 93, 94, 1995, 2002 Ian Lance Taylor\n",
	  VERSION);
  printf (" -s,--system system: Only consider files from named system\n");
  printf (" -x,--debug debug: Set debugging level\n");
#if HAVE_TAYLOR_CONFIG
  printf (" -I,--config file: Set configuration file to use\n");
#endif /* HAVE_TAYLOR_CONFIG */
  printf (" -v,--version: Print version and exit\n");
  printf (" --help: Print help and exit\n");
  printf ("Report bugs to taylor-uucp@gnu.org\n");
}

/* This routine is called by usysdep_walk_tree when moving the
   contents of an entire directory.  */

static void
upmovedir (zfull, zrelative, pinfo)
     const char *zfull;
     const char *zrelative;
     pointer pinfo;
{
  const char *ztodir = (const char *) pinfo;
  char *zto;

  zto = zsysdep_in_dir (ztodir, zrelative);
  if (zto == NULL)
    usysdep_exit (FALSE);
  upmove (zfull, zto);
  ubuffree (zto);
}

/* Move a file.  */

static void
upmove (zfrom, zto)
     const char *zfrom;
     const char *zto;
{
  (void) fsysdep_move_file (zfrom, zto, TRUE, TRUE, FALSE,
			    (const char *) NULL);
}
