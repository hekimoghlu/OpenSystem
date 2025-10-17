/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 20, 2023.
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
 * Include necessary headers.
 */

#include <cups/string-private.h>
#ifdef _WIN32
#  include <io.h>
#else
#  include <unistd.h>
#  include <fcntl.h>
#endif /* _WIN32 */

#include "ieee1284.c"


/*
 * 'main()' - Test the device-ID functions.
 */

int					/* O - Exit status */
main(int  argc,				/* I - Number of command-line args */
     char *argv[])			/* I - Command-line arguments */
{
  int	i,				/* Looping var */
	fd;				/* File descriptor */
  char	device_id[1024],		/* 1284 device ID string */
	make_model[1024],		/* make-and-model string */
	uri[1024];			/* URI string */


  if (argc < 2)
  {
    puts("Usage: test1284 device-file [... device-file-N]");
    exit(1);
  }

  for (i = 1; i < argc; i ++)
  {
    if ((fd = open(argv[i], O_RDWR)) < 0)
    {
      perror(argv[i]);
      return (errno);
    }

    printf("%s:\n", argv[i]);

    backendGetDeviceID(fd, device_id, sizeof(device_id), make_model,
                       sizeof(make_model), "test", uri, sizeof(uri));

    printf("    device_id=\"%s\"\n", device_id);
    printf("    make_model=\"%s\"\n", make_model);
    printf("    uri=\"%s\"\n", uri);

    close(fd);
  }

  return (0);
}
