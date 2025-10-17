/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 10, 2025.
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
/*
 * This file is 'mem-include-scan' clean, which means memdebug.h and
 * curl_memory.h are purposely not included in this file. See test 1132.
 *
 * The functions in this file are curlx functions which are not tracked by the
 * curl memory tracker memdebug.
 */

#include "curl_setup.h"

#if defined(_WIN32)

#include "curl_multibyte.h"

/*
 * MultiByte conversions using Windows kernel32 library.
 */

wchar_t *curlx_convert_UTF8_to_wchar(const char *str_utf8)
{
  wchar_t *str_w = NULL;

  if(str_utf8) {
    int str_w_len = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS,
                                        str_utf8, -1, NULL, 0);
    if(str_w_len > 0) {
      str_w = malloc(str_w_len * sizeof(wchar_t));
      if(str_w) {
        if(MultiByteToWideChar(CP_UTF8, 0, str_utf8, -1, str_w,
                               str_w_len) == 0) {
          free(str_w);
          return NULL;
        }
      }
    }
  }

  return str_w;
}

char *curlx_convert_wchar_to_UTF8(const wchar_t *str_w)
{
  char *str_utf8 = NULL;

  if(str_w) {
    int bytes = WideCharToMultiByte(CP_UTF8, 0, str_w, -1,
                                    NULL, 0, NULL, NULL);
    if(bytes > 0) {
      str_utf8 = malloc(bytes);
      if(str_utf8) {
        if(WideCharToMultiByte(CP_UTF8, 0, str_w, -1, str_utf8, bytes,
                               NULL, NULL) == 0) {
          free(str_utf8);
          return NULL;
        }
      }
    }
  }

  return str_utf8;
}

#endif /* _WIN32 */

#if defined(USE_WIN32_LARGE_FILES) || defined(USE_WIN32_SMALL_FILES)

int curlx_win32_open(const char *filename, int oflag, ...)
{
  int pmode = 0;

#ifdef _UNICODE
  int result = -1;
  wchar_t *filename_w = curlx_convert_UTF8_to_wchar(filename);
#endif

  va_list param;
  va_start(param, oflag);
  if(oflag & O_CREAT)
    pmode = va_arg(param, int);
  va_end(param);

#ifdef _UNICODE
  if(filename_w) {
    result = _wopen(filename_w, oflag, pmode);
    curlx_unicodefree(filename_w);
  }
  else
    errno = EINVAL;
  return result;
#else
  return (_open)(filename, oflag, pmode);
#endif
}

FILE *curlx_win32_fopen(const char *filename, const char *mode)
{
#ifdef _UNICODE
  FILE *result = NULL;
  wchar_t *filename_w = curlx_convert_UTF8_to_wchar(filename);
  wchar_t *mode_w = curlx_convert_UTF8_to_wchar(mode);
  if(filename_w && mode_w)
    result = _wfopen(filename_w, mode_w);
  else
    errno = EINVAL;
  curlx_unicodefree(filename_w);
  curlx_unicodefree(mode_w);
  return result;
#else
  return (fopen)(filename, mode);
#endif
}

int curlx_win32_stat(const char *path, struct_stat *buffer)
{
#ifdef _UNICODE
  int result = -1;
  wchar_t *path_w = curlx_convert_UTF8_to_wchar(path);
  if(path_w) {
#if defined(USE_WIN32_SMALL_FILES)
    result = _wstat(path_w, buffer);
#else
    result = _wstati64(path_w, buffer);
#endif
    curlx_unicodefree(path_w);
  }
  else
    errno = EINVAL;
  return result;
#else
#if defined(USE_WIN32_SMALL_FILES)
  return _stat(path, buffer);
#else
  return _stati64(path, buffer);
#endif
#endif
}

int curlx_win32_access(const char *path, int mode)
{
#if defined(_UNICODE)
  int result = -1;
  wchar_t *path_w = curlx_convert_UTF8_to_wchar(path);
  if(path_w) {
    result = _waccess(path_w, mode);
    curlx_unicodefree(path_w);
  }
  else
    errno = EINVAL;
  return result;
#else
  return _access(path, mode);
#endif
}

#endif /* USE_WIN32_LARGE_FILES || USE_WIN32_SMALL_FILES */
