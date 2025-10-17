/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 2, 2024.
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

#include "uudefs.h"
#include "sysdep.h"
#include "system.h"

/* Translate a uuto destination for Unix.  */

char *
zsysdep_uuto (zdest, zlocalname)
     const char *zdest;
     const char *zlocalname;
{
  const char *zexclam;
  char *zto;

  zexclam = strrchr (zdest, '!');
  if (zexclam == NULL)
    return NULL;
  zto = (char *) zbufalc (zexclam - zdest
			  + sizeof "!~/receive///"
			  + strlen (zexclam)
			  + strlen (zlocalname));
  memcpy (zto, zdest, (size_t) (zexclam - zdest));
  sprintf (zto + (zexclam - zdest), "!~/receive/%s/%s/",
	   zexclam + 1, zlocalname);
  return zto;
}
