/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 31, 2024.
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
/* Written by Bruno Haible <bruno@clisp.org>.  */

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Print an ASCII string with quotes and escape sequences where needed.  */
static void
print_escaped (stream, str)
     FILE *stream;
     const char *str;
{
  putc ('"', stream);
  for (; *str != '\0'; str++)
    if (*str == '\n')
      {
	fputs ("\\n\"", stream);
	if (str[1] == '\0')
	  return;
	fputs ("\n\"", stream);
      }
    else
      {
	if (*str == '"' || *str == '\\')
	  putc ('\\', stream);
	putc (*str, stream);
      }
  putc ('"', stream);
}

/* Add to the log file an entry denoting a failed translation.  */
void
_nl_log_untranslated (logfilename, domainname, msgid1, msgid2, plural)
     const char *logfilename;
     const char *domainname;
     const char *msgid1;
     const char *msgid2;
     int plural;
{
  static char *last_logfilename = NULL;
  static FILE *last_logfile = NULL;
  FILE *logfile;

  /* Can we reuse the last opened logfile?  */
  if (last_logfilename == NULL || strcmp (logfilename, last_logfilename) != 0)
    {
      /* Close the last used logfile.  */
      if (last_logfilename != NULL)
	{
	  if (last_logfile != NULL)
	    {
	      fclose (last_logfile);
	      last_logfile = NULL;
	    }
	  free (last_logfilename);
	  last_logfilename = NULL;
	}
      /* Open the logfile.  */
      last_logfilename = (char *) malloc (strlen (logfilename) + 1);
      if (last_logfilename == NULL)
	return;
      strcpy (last_logfilename, logfilename);
      last_logfile = fopen (logfilename, "a");
      if (last_logfile == NULL)
	return;
    }
  logfile = last_logfile;

  fprintf (logfile, "domain ");
  print_escaped (logfile, domainname);
  fprintf (logfile, "\nmsgid ");
  print_escaped (logfile, msgid1);
  if (plural)
    {
      fprintf (logfile, "\nmsgid_plural ");
      print_escaped (logfile, msgid2);
      fprintf (logfile, "\nmsgstr[0] \"\"\n");
    }
  else
    fprintf (logfile, "\nmsgstr \"\"\n");
  putc ('\n', logfile);
}
