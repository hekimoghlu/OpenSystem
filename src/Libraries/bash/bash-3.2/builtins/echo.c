/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 11, 2024.
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

#if defined (HAVE_UNISTD_H)
#  include <unistd.h>
#endif

#include "../bashansi.h"

#include <stdio.h>
#include "../shell.h"

#include "common.h"

#line 59 "echo.def"

#line 66 "echo.def"

#if defined (V9_ECHO)
#  define VALID_ECHO_OPTIONS "neE"
#else /* !V9_ECHO */
#  define VALID_ECHO_OPTIONS "n"
#endif /* !V9_ECHO */

/* System V machines already have a /bin/sh with a v9 behaviour.  We
   give Bash the identical behaviour for these machines so that the
   existing system shells won't barf.  Regrettably, the SUS v2 has
   standardized the Sys V echo behavior.  This variable is external
   so that we can have a `shopt' variable to control it at runtime. */
#if defined (DEFAULT_ECHO_TO_XPG) || defined (STRICT_POSIX)
int xpg_echo = 1;
#else
int xpg_echo = 0;
#endif /* DEFAULT_ECHO_TO_XPG */

extern int posixly_correct;

/* Print the words in LIST to standard output.  If the first word is
   `-n', then don't print a trailing newline.  We also support the
   echo syntax from Version 9 Unix systems. */
int
echo_builtin (list)
     WORD_LIST *list;
{
  int display_return, do_v9, i, len;
  char *temp, *s;

  do_v9 = xpg_echo || posixly_correct;
  display_return = 1;

  if (posixly_correct)
    goto just_echo;

  for (; list && (temp = list->word->word) && *temp == '-'; list = list->next)
    {
      /* If it appears that we are handling options, then make sure that
	 all of the options specified are actually valid.  Otherwise, the
	 string should just be echoed. */
      temp++;

      for (i = 0; temp[i]; i++)
	{
	  if (strchr (VALID_ECHO_OPTIONS, temp[i]) == 0)
	    break;
	}

      /* echo - and echo -<nonopt> both mean to just echo the arguments. */
      if (*temp == 0 || temp[i])
	break;

      /* All of the options in TEMP are valid options to ECHO.
	 Handle them. */
      while (i = *temp++)
	{
	  switch (i)
	    {
	    case 'n':
	      display_return = 0;
	      break;
#if defined (V9_ECHO)
	    case 'e':
	      do_v9 = 1;
	      break;
	    case 'E':
	      do_v9 = 0;
	      break;
#endif /* V9_ECHO */
	    default:
	      goto just_echo;	/* XXX */
	    }
	}
    }

just_echo:

  clearerr (stdout);	/* clear error before writing and testing success */

  while (list)
    {
      i = len = 0;
      temp = do_v9 ? ansicstr (list->word->word, STRLEN (list->word->word), 1, &i, &len)
		   : list->word->word;
      if (temp)
	{
	  if (do_v9)
	    {
	      for (s = temp; len > 0; len--)
		putchar (*s++);
	    }
	  else	    
	    printf ("%s", temp);
#if defined (SunOS5)
	  fflush (stdout);	/* Fix for bug in SunOS 5.5 printf(3) */
#endif
	}
      if (do_v9 && temp)
	free (temp);
      list = list->next;
      if (i)
	{
	  display_return = 0;
	  break;
	}
      if (list)
	putchar(' ');
    }

  if (display_return)
    putchar ('\n');
  fflush (stdout);
  if (ferror (stdout))
    {
      sh_wrerror ();
      clearerr (stdout);
      return (EXECUTION_FAILURE);
    }
  return (EXECUTION_SUCCESS);
}
