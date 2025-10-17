/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 19, 2023.
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

#include "adminutil.h"
#include "string-private.h"


/*
 * Local functions...
 */

static void	show_settings(int num_settings, cups_option_t *settings);


/*
 * 'main()' - Main entry.
 */

int					/* O - Exit status */
main(int  argc,				/* I - Number of command-line args */
     char *argv[])			/* I - Command-line arguments */
{
  int		i,			/* Looping var */
		num_settings;		/* Number of settings */
  cups_option_t	*settings;		/* Settings */
  http_t	*http;			/* Connection to server */


 /*
  * Connect to the server using the defaults...
  */

  http = httpConnect2(cupsServer(), ippPort(), NULL, AF_UNSPEC,
                      cupsEncryption(), 1, 30000, NULL);

 /*
  * Set the current configuration if we have anything on the command-line...
  */

  if (argc > 1)
  {
    for (i = 1, num_settings = 0, settings = NULL; i < argc; i ++)
      num_settings = cupsParseOptions(argv[i], num_settings, &settings);

    if (cupsAdminSetServerSettings(http, num_settings, settings))
    {
      puts("New server settings:");
      cupsFreeOptions(num_settings, settings);
    }
    else
    {
      printf("Server settings not changed: %s\n", cupsLastErrorString());
      return (1);
    }
  }
  else
    puts("Current server settings:");

 /*
  * Get the current configuration...
  */

  if (cupsAdminGetServerSettings(http, &num_settings, &settings))
  {
    show_settings(num_settings, settings);
    cupsFreeOptions(num_settings, settings);
    return (0);
  }
  else
  {
    printf("    %s\n", cupsLastErrorString());
    return (1);
  }
}


/*
 * 'show_settings()' - Show settings in the array...
 */

static void
show_settings(
    int           num_settings,		/* I - Number of settings */
    cups_option_t *settings)		/* I - Settings */
{
  while (num_settings > 0)
  {
    printf("    %s=%s\n", settings->name, settings->value);

    settings ++;
    num_settings --;
  }
}
