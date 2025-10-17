/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 4, 2024.
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
#include "curl_setup.h"

/*
 * we force our own host name, in order to make some tests machine independent
 */

int gethostname(char *name, GETHOSTNAME_TYPE_ARG2 namelen)
{
  const char *force_hostname = getenv("CURL_GETHOSTNAME");
  if(force_hostname) {
    strncpy(name, force_hostname, namelen);
    name[namelen-1] = '\0';
    return 0;
  }

  /* LD_PRELOAD used, but no hostname set, we'll just return a failure */
  return -1;
}
