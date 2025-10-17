/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 1, 2023.
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

#if defined(HAVE_STRCASECMP) && defined(HAVE_STRINGS_H)
#include <strings.h>
#endif

#include "tool_util.h"

#include "memdebug.h" /* keep this as LAST include */

#if defined(_WIN32)

/* In case of bug fix this function has a counterpart in timeval.c */
struct timeval tvnow(void)
{
  struct timeval now;
  if(tool_isVistaOrGreater) { /* QPC timer might have issues pre-Vista */
    LARGE_INTEGER count;
    QueryPerformanceCounter(&count);
    now.tv_sec = (long)(count.QuadPart / tool_freq.QuadPart);
    now.tv_usec = (long)((count.QuadPart % tool_freq.QuadPart) * 1000000 /
                         tool_freq.QuadPart);
  }
  else {
    /* Disable /analyze warning that GetTickCount64 is preferred  */
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable:28159)
#endif
    DWORD milliseconds = GetTickCount();
#if defined(_MSC_VER)
#pragma warning(pop)
#endif

    now.tv_sec = (long)(milliseconds / 1000);
    now.tv_usec = (long)((milliseconds % 1000) * 1000);
  }
  return now;
}

#elif defined(HAVE_CLOCK_GETTIME_MONOTONIC)

struct timeval tvnow(void)
{
  /*
  ** clock_gettime() is granted to be increased monotonically when the
  ** monotonic clock is queried. Time starting point is unspecified, it
  ** could be the system start-up time, the Epoch, or something else,
  ** in any case the time starting point does not change once that the
  ** system has started up.
  */
  struct timeval now;
  struct timespec tsnow;
  if(0 == clock_gettime(CLOCK_MONOTONIC, &tsnow)) {
    now.tv_sec = tsnow.tv_sec;
    now.tv_usec = (int)(tsnow.tv_nsec / 1000);
  }
  /*
  ** Even when the configure process has truly detected monotonic clock
  ** availability, it might happen that it is not actually available at
  ** run-time. When this occurs simply fallback to other time source.
  */
#ifdef HAVE_GETTIMEOFDAY
  else
    (void)gettimeofday(&now, NULL);
#else
  else {
    now.tv_sec = time(NULL);
    now.tv_usec = 0;
  }
#endif
  return now;
}

#elif defined(HAVE_GETTIMEOFDAY)

struct timeval tvnow(void)
{
  /*
  ** gettimeofday() is not granted to be increased monotonically, due to
  ** clock drifting and external source time synchronization it can jump
  ** forward or backward in time.
  */
  struct timeval now;
  (void)gettimeofday(&now, NULL);
  return now;
}

#else

struct timeval tvnow(void)
{
  /*
  ** time() returns the value of time in seconds since the Epoch.
  */
  struct timeval now;
  now.tv_sec = time(NULL);
  now.tv_usec = 0;
  return now;
}

#endif

/*
 * Make sure that the first argument is the more recent time, as otherwise
 * we'll get a weird negative time-diff back...
 *
 * Returns: the time difference in number of milliseconds.
 */
long tvdiff(struct timeval newer, struct timeval older)
{
  return (long)(newer.tv_sec-older.tv_sec)*1000+
    (long)(newer.tv_usec-older.tv_usec)/1000;
}

/* Case insensitive compare. Accept NULL pointers. */
int struplocompare(const char *p1, const char *p2)
{
  if(!p1)
    return p2? -1: 0;
  if(!p2)
    return 1;
#ifdef HAVE_STRCASECMP
  return strcasecmp(p1, p2);
#elif defined(HAVE_STRCMPI)
  return strcmpi(p1, p2);
#elif defined(HAVE_STRICMP)
  return stricmp(p1, p2);
#else
  return strcmp(p1, p2);
#endif
}

/* Indirect version to use as qsort callback. */
int struplocompare4sort(const void *p1, const void *p2)
{
  return struplocompare(* (char * const *) p1, * (char * const *) p2);
}
