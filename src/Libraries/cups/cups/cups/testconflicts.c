/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 26, 2022.
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
/*
 * Include necessary headers...
 */

#include "cups.h"
#include "ppd.h"
#include "string-private.h"


/*
 * 'main()' - Main entry.
 */

int					/* O - Exit status */
main(int  argc,				/* I - Number of command-line arguments */
     char *argv[])			/* I - Command-line arguments */
{
  int		i;			/* Looping var */
  ppd_file_t	*ppd;			/* PPD file loaded from disk */
  char		line[256],		/* Input buffer */
		*ptr,			/* Pointer into buffer */
		*optr,			/* Pointer to first option name */
		*cptr;			/* Pointer to first choice */
  int		num_options;		/* Number of options */
  cups_option_t	*options;		/* Options */
  char		*option,		/* Current option */
		*choice;		/* Current choice */


  if (argc != 2)
  {
    puts("Usage: testconflicts filename.ppd");
    return (1);
  }

  if ((ppd = ppdOpenFile(argv[1])) == NULL)
  {
    ppd_status_t	err;		/* Last error in file */
    int			linenum;	/* Line number in file */

    err = ppdLastError(&linenum);

    printf("Unable to open PPD file \"%s\": %s on line %d\n", argv[1],
           ppdErrorString(err), linenum);
    return (1);
  }

  ppdMarkDefaults(ppd);

  option = NULL;
  choice = NULL;

  for (;;)
  {
    num_options = 0;
    options     = NULL;

    if (!cupsResolveConflicts(ppd, option, choice, &num_options, &options))
      puts("Unable to resolve conflicts!");
    else if ((!option && num_options > 0) || (option && num_options > 1))
    {
      fputs("Resolved conflicts with the following options:\n   ", stdout);
      for (i = 0; i < num_options; i ++)
        if (!option || _cups_strcasecmp(option, options[i].name))
	  printf(" %s=%s", options[i].name, options[i].value);
      putchar('\n');

      cupsFreeOptions(num_options, options);
    }

    if (option)
    {
      free(option);
      option = NULL;
    }

    if (choice)
    {
      free(choice);
      choice = NULL;
    }

    printf("\nNew Option(s): ");
    fflush(stdout);
    if (!fgets(line, sizeof(line), stdin) || line[0] == '\n')
      break;

    for (ptr = line; isspace(*ptr & 255); ptr ++);
    for (optr = ptr; *ptr && *ptr != '='; ptr ++);
    if (!*ptr)
      break;
    for (*ptr++ = '\0', cptr = ptr; *ptr && !isspace(*ptr & 255); ptr ++);
    if (!*ptr)
      break;
    *ptr++ = '\0';

    option      = strdup(optr);
    choice      = strdup(cptr);
    num_options = cupsParseOptions(ptr, 0, &options);

    ppdMarkOption(ppd, option, choice);
    if (cupsMarkOptions(ppd, num_options, options))
      puts("Options Conflict!");
    cupsFreeOptions(num_options, options);
  }

  if (option)
    free(option);
  if (choice)
    free(choice);

  return (0);
}
