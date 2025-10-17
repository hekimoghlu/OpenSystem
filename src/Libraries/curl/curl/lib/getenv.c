/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 30, 2024.
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

#include <curl/curl.h>
#include "curl_memory.h"

#include "memdebug.h"

static char *GetEnv(const char *variable)
{
#if defined(_WIN32_WCE) || defined(CURL_WINDOWS_APP) || \
  defined(__ORBIS__) || defined(__PROSPERO__) /* PlayStation 4 and 5 */
  (void)variable;
  return NULL;
#elif defined(_WIN32)
  /* This uses Windows API instead of C runtime getenv() to get the environment
     variable since some changes aren't always visible to the latter. #4774 */
  char *buf = NULL;
  char *tmp;
  DWORD bufsize;
  DWORD rc = 1;
  const DWORD max = 32768; /* max env var size from MSCRT source */

  for(;;) {
    tmp = realloc(buf, rc);
    if(!tmp) {
      free(buf);
      return NULL;
    }

    buf = tmp;
    bufsize = rc;

    /* It's possible for rc to be 0 if the variable was found but empty.
       Since getenv doesn't make that distinction we ignore it as well. */
    rc = GetEnvironmentVariableA(variable, buf, bufsize);
    if(!rc || rc == bufsize || rc > max) {
      free(buf);
      return NULL;
    }

    /* if rc < bufsize then rc is bytes written not including null */
    if(rc < bufsize)
      return buf;

    /* else rc is bytes needed, try again */
  }
#else
  char *env = getenv(variable);
  return (env && env[0])?strdup(env):NULL;
#endif
}

char *curl_getenv(const char *v)
{
  return GetEnv(v);
}
