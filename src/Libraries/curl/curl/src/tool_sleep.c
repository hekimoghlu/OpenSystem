/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 12, 2023.
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
#include "tool_setup.h"

#ifdef HAVE_SYS_SELECT_H
#  include <sys/select.h>
#elif defined(HAVE_UNISTD_H)
#  include <unistd.h>
#endif

#ifdef HAVE_POLL_H
#  include <poll.h>
#elif defined(HAVE_SYS_POLL_H)
#  include <sys/poll.h>
#endif

#ifdef MSDOS
#  include <dos.h>
#endif

#include "tool_sleep.h"

#include "memdebug.h" /* keep this as LAST include */

void tool_go_sleep(long ms)
{
#if defined(MSDOS)
  delay(ms);
#elif defined(_WIN32)
  Sleep(ms);
#elif defined(HAVE_POLL_FINE)
  (void)poll((void *)0, 0, (int)ms);
#else
  struct timeval timeout;
  timeout.tv_sec = ms / 1000L;
  ms = ms % 1000L;
  timeout.tv_usec = (int)ms * 1000;
  select(0, NULL,  NULL, NULL, &timeout);
#endif
}
