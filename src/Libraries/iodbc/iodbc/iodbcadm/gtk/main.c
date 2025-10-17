/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 27, 2023.
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
#include <iodbc.h>
#include <isql.h>
#include <odbcinst.h>
#include <unistd.h>
#include <stdlib.h>

#include "gui.h"


int
gtk_gui (int *argc, char **argv[])
{
  GtkWidget *mainwnd;
#if GTK_CHECK_VERSION(2,0,0)
  gtk_set_locale();
#endif
  gtk_init (argc, argv);
  mainwnd = gtk_window_new (GTK_WINDOW_TOPLEVEL);
  return SQLManageDataSources (mainwnd);
}


int
kde_gui (int *argc, char **argv[])
{
  return -1;
}


void
display_help (void)
{
  printf ("-help\t\t\tDisplay the list of options.\n\r");
  printf
      ("-odbc filename\t\tSet the location of the user ODBC.INI file.\n\r");
  printf
      ("-odbcinst filename\tSet the location of the user ODBCINST.INI file.\n\r");
  printf
      ("-sysodbc filename\tSet the location of the system ODBC.INI file.\n\r");
  printf
      ("-sysodbcinst filename\tSet the location of the system ODBCINST.INI file.\n\r");
  printf ("-gui guitype\t\tSet the GUI type : GTK, KDE.\n\r");
  printf ("-debug\t\t\tThe error messages are displayed on the console.\n\r");
  printf
      ("-admin odbcinstfile\tUsed to administrate the system odbcinst.ini file.\n\r\n\r");
  _exit (1);
}


#if !defined(HAVE_SETENV)
static int
setenv (const char *name, const char *value, int overwrite)
{
  int rc;
  char *entry;

  /*
   *  Allocate some space for new environment variable
   */
  if ((entry = (char *) malloc (strlen (name) + strlen (value) + 2)) == NULL)
    return -1;
  strcpy (entry, name);
  strcat (entry, "=");
  strcat (entry, value);

  /*
   *  Check if variable already exists in current environment and whether
   *  we want to overwrite it with a new value if it exists.
   */
  if (getenv (name) != NULL && !overwrite)
    {
      free (entry);
      return 0;
    }

  /*
   *  Add the variable to the environment.
   */
  rc = putenv (entry);
  free (entry);
  return (rc == 0) ? 0 : -1;
}
#endif /* HAVE_SETENV */


int
main (int argc, char *argv[])
{
  BOOL debug = FALSE;
  char path[4096];
  char *gui = NULL;
  int i = 1;

  printf ("iODBC Administrator (GTK)\n");
  printf ("%s\n", PACKAGE_STRING);
  printf ("Copyright (C) 2000-2006 OpenLink Software\n");
  printf ("Please report all bugs to <%s>\n\n", PACKAGE_BUGREPORT);

  /* Check options commands */
  if (argc > 1)
    {
      for (; i < argc; i++)
	{
	  if (!strcasecmp (argv[i], "-help"))
	    display_help ();

	  if (!strcasecmp (argv[i], "-debug"))
	    debug = TRUE;

	  if (!strcasecmp (argv[i], "-odbc"))
	    {
	      if (i + 1 >= argc)
		display_help ();
	      setenv ("ODBCINI", argv[++i], TRUE);
	    }

	  if (!strcasecmp (argv[i], "-admin"))
	    {
	      if (i + 1 >= argc)
		display_help ();
	      setenv ("ODBCINSTINI", argv[++i], TRUE);
	      setenv ("SYSODBCINSTINI", argv[i], TRUE);
	    }

	  if (!strcasecmp (argv[i], "-odbcinst"))
	    {
	      if (i + 1 >= argc)
		display_help ();
	      setenv ("ODBCINSTINI", argv[++i], TRUE);
	    }

	  if (!strcasecmp (argv[i], "-sysodbc"))
	    {
	      if (i + 1 >= argc)
		display_help ();
	      setenv ("SYSODBCINI", argv[++i], TRUE);
	    }

	  if (!strcasecmp (argv[i], "-sysodbcinst"))
	    {
	      if (i + 1 >= argc)
		display_help ();
	      setenv ("SYSODBCINSTINI", argv[++i], TRUE);
	    }

	  if (!strcasecmp (argv[i], "-gui"))
	    {
	      if (i + 2 >= argc)
		display_help ();
	      gui = argv[++i];
	    }
	}
    }

  if (!getenv ("ODBCINI") && getenv ("HOME"))
    {
      STRCPY (path, getenv ("HOME"));
      STRCAT (path, "/.odbc.ini");
      setenv ("ODBCINI", path, TRUE);
    }

  if (!debug)
    {
      close (STDOUT_FILENO);
      close (STDERR_FILENO);
    }

  if (gui && !strcasecmp (gui, "KDE"))
    return kde_gui (&argc, &argv);

  return gtk_gui (&argc, &argv);
}
