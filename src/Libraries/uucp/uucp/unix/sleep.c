/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 29, 2025.
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
#include "system.h"

void
usysdep_sleep (c)
     int c;
{
#if HAVE_NAPMS || HAVE_NAP || HAVE_USLEEP || HAVE_SELECT || HAVE_POLL
  int i;

  /* In this case, usysdep_pause is accurate.  */
  for (i = 2 * c; i > 0; i--)
    usysdep_pause ();
#else
  /* On some system sleep (1) may not sleep at all.  Avoid this sort
     of problem by always doing at least sleep (2).  */
  if (c < 2)
    c = 2;
  (void) sleep (c);
#endif
}
