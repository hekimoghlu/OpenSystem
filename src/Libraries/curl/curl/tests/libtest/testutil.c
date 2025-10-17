/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 23, 2023.
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
#include "testutil.h"
#include "memdebug.h"

#if defined(_WIN32)

struct timeval tutil_tvnow(void)
{
  /*
  ** GetTickCount() is available on _all_ Windows versions from W95 up
  ** to nowadays. Returns milliseconds elapsed since last system boot,
  ** increases monotonically and wraps once 49.7 days have elapsed.
  */
  struct timeval now;
  DWORD milliseconds = GetTickCount();
  now.tv_sec = (long)(milliseconds / 1000);
  now.tv_usec = (long)((milliseconds % 1000) * 1000);
  return now;
}

#elif defined(HAVE_CLOCK_GETTIME_MONOTONIC)

struct timeval tutil_tvnow(void)
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

struct timeval tutil_tvnow(void)
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

struct timeval tutil_tvnow(void)
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
long tutil_tvdiff(struct timeval newer, struct timeval older)
{
  return (long)(newer.tv_sec-older.tv_sec)*1000+
    (long)(newer.tv_usec-older.tv_usec)/1000;
}

/*
 * Same as tutil_tvdiff but with full usec resolution.
 *
 * Returns: the time difference in seconds with subsecond resolution.
 */
double tutil_tvdiff_secs(struct timeval newer, struct timeval older)
{
  if(newer.tv_sec != older.tv_sec)
    return (double)(newer.tv_sec-older.tv_sec)+
      (double)(newer.tv_usec-older.tv_usec)/1000000.0;
  return (double)(newer.tv_usec-older.tv_usec)/1000000.0;
}

#ifdef _WIN32
HMODULE win32_load_system_library(const TCHAR *filename)
{
  size_t filenamelen = _tcslen(filename);
  size_t systemdirlen = GetSystemDirectory(NULL, 0);
  size_t written;
  TCHAR *path;

  if(!filenamelen || filenamelen > 32768 ||
     !systemdirlen || systemdirlen > 32768)
    return NULL;

  /* systemdirlen includes null character */
  path = malloc(sizeof(TCHAR) * (systemdirlen + 1 + filenamelen));
  if(!path)
    return NULL;

  /* if written >= systemdirlen then nothing was written */
  written = GetSystemDirectory(path, (unsigned int)systemdirlen);
  if(!written || written >= systemdirlen)
    return NULL;

  if(path[written - 1] != _T('\\'))
    path[written++] = _T('\\');

  _tcscpy(path + written, filename);

  return LoadLibrary(path);
}
#endif
