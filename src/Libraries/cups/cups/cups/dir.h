/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 17, 2021.
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
#ifndef _CUPS_DIR_H_
#  define _CUPS_DIR_H_


/*
 * Include necessary headers...
 */

#  include "versioning.h"
#  include <sys/stat.h>


/*
 * C++ magic...
 */

#  ifdef __cplusplus
extern "C" {
#  endif /* __cplusplus */


/*
 * Data types...
 */

typedef struct _cups_dir_s cups_dir_t;	/**** Directory type ****/

typedef struct cups_dentry_s		/**** Directory entry type ****/
{
  char		filename[260];		/* File name */
  struct stat	fileinfo;		/* File information */
} cups_dentry_t;


/*
 * Prototypes...
 */

extern void		cupsDirClose(cups_dir_t *dp) _CUPS_API_1_2;
extern cups_dir_t	*cupsDirOpen(const char *directory) _CUPS_API_1_2;
extern cups_dentry_t	*cupsDirRead(cups_dir_t *dp) _CUPS_API_1_2;
extern void		cupsDirRewind(cups_dir_t *dp) _CUPS_API_1_2;


#  ifdef __cplusplus
}
#  endif /* __cplusplus */
#endif /* !_CUPS_DIR_H_ */
