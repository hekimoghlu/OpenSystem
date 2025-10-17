/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 22, 2023.
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

/* See whether the user is privileged (for example, only privileged
   users are permitted to kill arbitrary jobs with uustat).  This is
   true only for root and uucp.  We check for uucp by seeing if the
   real user ID and the effective user ID are the same; this works
   because we should be suid to uucp, so our effective user ID will
   always be uucp while our real user ID will be whoever ran the
   program.  */

boolean
fsysdep_privileged ()
{
  uid_t iuid;

  iuid = getuid ();
  return iuid == 0 || iuid == geteuid ();
}
