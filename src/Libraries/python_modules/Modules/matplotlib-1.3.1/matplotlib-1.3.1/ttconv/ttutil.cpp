/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 6, 2023.
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
 * Modified for use within matplotlib
 * 5 July 2007
 * Michael Droettboom
 */

/* Very simple interface to the ppr TT routines */
/* (c) Frank Siegert 1996 */

#include "global_defines.h"
#include <cstdio>
#include <cstdarg>
#include <cstdlib>
#include "pprdrv.h"

#if DEBUG_TRUETYPE
void debug(const char *format, ... )
{
  va_list arg_list;
  va_start(arg_list, format);

  printf(format, arg_list);

  va_end(arg_list);
}
#endif

#define PRINTF_BUFFER_SIZE 512
void TTStreamWriter::printf(const char* format, ...)
{
  va_list arg_list;
  va_start(arg_list, format);
  char buffer[PRINTF_BUFFER_SIZE];

#if defined(WIN32) || defined(_MSC_VER)
  int size = _vsnprintf(buffer, PRINTF_BUFFER_SIZE, format, arg_list);
#else
  int size = vsnprintf(buffer, PRINTF_BUFFER_SIZE, format, arg_list);
#endif
  if (size >= PRINTF_BUFFER_SIZE) {
    char* buffer2 = (char*)malloc(size);
#if defined(WIN32) || defined(_MSC_VER)
    _vsnprintf(buffer2, size, format, arg_list);
#else
    vsnprintf(buffer2, size, format, arg_list);
#endif
    free(buffer2);
  } else {
    this->write(buffer);
  }

  va_end(arg_list);
}

void TTStreamWriter::put_char(int val)
{
  char c[2];
  c[0] = (char)val;
  c[1] = 0;
  this->write(c);
}

void TTStreamWriter::puts(const char *a)
{
  this->write(a);
}

void TTStreamWriter::putline(const char *a)
{
  this->write(a);
  this->write("\n");
}

void replace_newlines_with_spaces(char *a) {
  char* i = a;
  while (*i != 0) {
    if (*i == '\n')
      *i = ' ';
    i++;
  }
}
