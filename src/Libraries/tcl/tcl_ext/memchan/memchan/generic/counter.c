/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 20, 2023.
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
#include "memchanInt.h"

Tcl_Obj*
MemchanGenHandle (prefix)
CONST char* prefix;
{
  /* 3 alternatives for implementation:
   * a) Tcl before 8.x
   * b) 8.0.x (objects, non-threaded)
   * c) 8.1.x (objects, possibly threaded)
   */

  /*
   * count number of generated (memory) channels,
   * used for id generation. Ids are never reclaimed
   * and there is no dealing with wrap around. On the
   * other hand, "unsigned long" should be big enough
   * except for absolute longrunners (generate a 100 ids
   * per second => overflow will occur in 1 1/3 years).
   */

#if GT81
  TCL_DECLARE_MUTEX (memchanCounterMutex)
  static unsigned long memCounter = 0;

  char     channelName [50];
  Tcl_Obj* res = Tcl_NewStringObj ((char*) prefix, -1);

  Tcl_MutexLock (&memchanCounterMutex);
  {
    LTOA (memCounter, channelName);
    memCounter ++;
  }
  Tcl_MutexUnlock (&memchanCounterMutex);

  Tcl_AppendStringsToObj (res, channelName, (char*) NULL);

  return res;

#else /* TCL_MAJOR_VERSION == 8 */
  static unsigned long memCounter = 0;

  char     channelName [50];
  Tcl_Obj* res = Tcl_NewStringObj ((char*) prefix, -1);

  LTOA (memCounter, channelName);
  memCounter ++;

  Tcl_AppendStringsToObj (res, channelName, (char*) NULL);

  return res;
#endif
}
