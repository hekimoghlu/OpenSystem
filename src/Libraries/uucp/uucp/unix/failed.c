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
#include "uucp.h"

#include "sysdep.h"
#include "uudefs.h"
#include "system.h"

char *
zsysdep_save_failed_file (zfile)
     const char *zfile;
{
  char *zto;

  zto = zsappend3 (zSspooldir, FAILEDDIR, zfile);

  if (! fsysdep_move_file (zfile, zto, TRUE, FALSE, FALSE,
			   (const char *) NULL))
    {
      ubuffree (zto);
      return NULL;
    }

  return zto;
}
