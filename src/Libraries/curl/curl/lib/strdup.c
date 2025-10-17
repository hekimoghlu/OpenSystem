/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 28, 2024.
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

#ifdef _WIN32
#include <wchar.h>
#endif

#include "strdup.h"
#include "curl_memory.h"

/* The last #include file should be: */
#include "memdebug.h"

#ifndef HAVE_STRDUP
char *Curl_strdup(const char *str)
{
  size_t len;
  char *newstr;

  if(!str)
    return (char *)NULL;

  len = strlen(str) + 1;

  newstr = malloc(len);
  if(!newstr)
    return (char *)NULL;

  memcpy(newstr, str, len);
  return newstr;
}
#endif

#ifdef _WIN32
/***************************************************************************
 *
 * Curl_wcsdup(source)
 *
 * Copies the 'source' wchar string to a newly allocated buffer (that is
 * returned).
 *
 * Returns the new pointer or NULL on failure.
 *
 ***************************************************************************/
wchar_t *Curl_wcsdup(const wchar_t *src)
{
  size_t length = wcslen(src);

  if(length > (SIZE_T_MAX / sizeof(wchar_t)) - 1)
    return (wchar_t *)NULL; /* integer overflow */

  return (wchar_t *)Curl_memdup(src, (length + 1) * sizeof(wchar_t));
}
#endif

/***************************************************************************
 *
 * Curl_memdup(source, length)
 *
 * Copies the 'source' data to a newly allocated buffer (that is
 * returned). Copies 'length' bytes.
 *
 * Returns the new pointer or NULL on failure.
 *
 ***************************************************************************/
void *Curl_memdup(const void *src, size_t length)
{
  void *buffer = malloc(length);
  if(!buffer)
    return NULL; /* fail */

  memcpy(buffer, src, length);

  return buffer;
}

/***************************************************************************
 *
 * Curl_memdup0(source, length)
 *
 * Copies the 'source' string to a newly allocated buffer (that is returned).
 * Copies 'length' bytes then adds a null terminator.
 *
 * Returns the new pointer or NULL on failure.
 *
 ***************************************************************************/
void *Curl_memdup0(const char *src, size_t length)
{
  char *buf = malloc(length + 1);
  if(!buf)
    return NULL;
  memcpy(buf, src, length);
  buf[length] = 0;
  return buf;
}

/***************************************************************************
 *
 * Curl_saferealloc(ptr, size)
 *
 * Does a normal realloc(), but will free the data pointer if the realloc
 * fails. If 'size' is non-zero, it will free the data and return a failure.
 *
 * This convenience function is provided and used to help us avoid a common
 * mistake pattern when we could pass in a zero, catch the NULL return and end
 * up free'ing the memory twice.
 *
 * Returns the new pointer or NULL on failure.
 *
 ***************************************************************************/
void *Curl_saferealloc(void *ptr, size_t size)
{
  void *datap = realloc(ptr, size);
  if(size && !datap)
    /* only free 'ptr' if size was non-zero */
    free(ptr);
  return datap;
}
