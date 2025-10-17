/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 1, 2022.
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
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 27, 2022.
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
 */#include <sys/types.h>
#include <sys/file.h>
#include <sys/disk.h>
#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

char *blockcheck __P((char *));

long dksecsize (dev)
     char *dev;
{
    int    fd;        /* file descriptor for reading device label */
    char   *cdev;
    int    devblklen;
    extern int errno;

    /* Convert block device into a character device string */
    if ((cdev = blockcheck(dev))) {
        if ((fd = open(cdev, O_RDONLY)) < 0) {
  	  fprintf(stderr, "Can't open %s, %s\n", cdev, strerror(errno));
	  return (0);
        }
    }
    else
          return (0);

    if (ioctl(fd, DKIOCGETBLOCKSIZE, &devblklen) < 0) {
	(void)close(fd);
        return (0);
    }
    else {
	(void)close(fd);
        return (devblklen);
    }
}


