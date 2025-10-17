/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 21, 2024.
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
#include <stdio.h>

#ifdef READLINE_LIBRARY
#  include "history.h"
#else
#  include <readline/history.h>
#endif

#include <string.h>

main (argc, argv)
     int argc;
     char **argv;
{
  char line[1024], *t;
  int len, done;

  line[0] = 0;
  done = 0;

  using_history ();
  while (!done)
    {
      printf ("history$ ");
      fflush (stdout);
      t = fgets (line, sizeof (line) - 1, stdin);
      if (t && *t)
	{
	  len = strlen (t);
	  if (t[len - 1] == '\n')
	    t[len - 1] = '\0';
	}

      if (!t)
	strcpy (line, "quit");

      if (line[0])
	{
	  char *expansion;
	  int result;

	  using_history ();

	  result = history_expand (line, &expansion);
	  if (result)
	    fprintf (stderr, "%s\n", expansion);

	  if (result < 0 || result == 2)
	    {
	      free (expansion);
	      continue;
	    }

	  add_history (expansion);
	  strncpy (line, expansion, sizeof (line) - 1);
	  free (expansion);
	}

      if (strcmp (line, "quit") == 0)
	done = 1;
      else if (strcmp (line, "save") == 0)
	write_history ("history_file");
      else if (strcmp (line, "read") == 0)
	read_history ("history_file");
      else if (strcmp (line, "list") == 0)
	{
	  register HIST_ENTRY **the_list;
	  register int i;
	  time_t tt;
	  char timestr[128];

	  the_list = history_list ();
	  if (the_list)
	    for (i = 0; the_list[i]; i++)
	      {
	      	tt = history_get_time (the_list[i]);
		if (tt)
		  strftime (timestr, sizeof (timestr), "%a %R", localtime(&tt));
		else
		  strcpy (timestr, "??");
	        printf ("%d: %s: %s\n", i + history_base, timestr, the_list[i]->line);
	      }
	}
      else if (strncmp (line, "delete", 6) == 0)
	{
	  int which;
	  if ((sscanf (line + 6, "%d", &which)) == 1)
	    {
	      HIST_ENTRY *entry = remove_history (which);
	      if (!entry)
		fprintf (stderr, "No such entry %d\n", which);
	      else
		{
		  free (entry->line);
		  free (entry);
		}
	    }
	  else
	    {
	      fprintf (stderr, "non-numeric arg given to `delete'\n");
	    }
	}
    }
}
