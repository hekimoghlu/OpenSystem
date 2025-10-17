/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 13, 2023.
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
#include "rename.h"

#include "curl_setup.h"

#if (!defined(CURL_DISABLE_HTTP) || !defined(CURL_DISABLE_COOKIES)) || \
  !defined(CURL_DISABLE_ALTSVC)

#include "curl_multibyte.h"
#include "timeval.h"

/* The last 3 #include files should be in this order */
#include "curl_printf.h"
#include "curl_memory.h"
#include "memdebug.h"

/* return 0 on success, 1 on error */
int Curl_rename(const char *oldpath, const char *newpath)
{
#ifdef _WIN32
  /* rename() on Windows doesn't overwrite, so we can't use it here.
     MoveFileEx() will overwrite and is usually atomic, however it fails
     when there are open handles to the file. */
  const int max_wait_ms = 1000;
  struct curltime start = Curl_now();
  TCHAR *tchar_oldpath = curlx_convert_UTF8_to_tchar((char *)oldpath);
  TCHAR *tchar_newpath = curlx_convert_UTF8_to_tchar((char *)newpath);
  for(;;) {
    timediff_t diff;
    if(MoveFileEx(tchar_oldpath, tchar_newpath, MOVEFILE_REPLACE_EXISTING)) {
      curlx_unicodefree(tchar_oldpath);
      curlx_unicodefree(tchar_newpath);
      break;
    }
    diff = Curl_timediff(Curl_now(), start);
    if(diff < 0 || diff > max_wait_ms) {
      curlx_unicodefree(tchar_oldpath);
      curlx_unicodefree(tchar_newpath);
      return 1;
    }
    Sleep(1);
  }
#else
  if(rename(oldpath, newpath))
    return 1;
#endif
  return 0;
}

#endif
