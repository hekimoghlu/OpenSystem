/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 30, 2022.
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
#include <stdio_ext.h>

#include <errno.h>
#include <stdlib.h>

#include <async_safe/log.h>

#include "local.h"

size_t __fbufsize(FILE* fp) {
  return fp->_bf._size;
}

int __freading(FILE* fp) {
  return (fp->_flags & __SRD) != 0;
}

int __fwriting(FILE* fp) {
  return (fp->_flags & __SWR) != 0;
}

int __freadable(FILE* fp) {
  return (fp->_flags & (__SRD|__SRW)) != 0;
}

int __fwritable(FILE* fp) {
  return (fp->_flags & (__SWR|__SRW)) != 0;
}

int __flbf(FILE* fp) {
  return (fp->_flags & __SLBF) != 0;
}

size_t __fpending(FILE* fp) {
  return fp->_p - fp->_bf._base;
}

size_t __freadahead(FILE* fp) {
  // Normally _r is the amount of input already available.
  // When there's ungetc() data, _r counts that and _ur is the previous _r.
  return fp->_r + (HASUB(fp) ? fp->_ur : 0);
}

void _flushlbf() {
  // If we flush all streams, we know we've flushed all the line-buffered streams.
  fflush(nullptr);
}

void __fseterr(FILE* fp) {
  fp->_flags |= __SERR;
}

int __fsetlocking(FILE* fp, int type) {
  int old_state = _EXT(fp)->_caller_handles_locking ? FSETLOCKING_BYCALLER : FSETLOCKING_INTERNAL;
  if (type == FSETLOCKING_QUERY) {
    return old_state;
  }

  if (type != FSETLOCKING_INTERNAL && type != FSETLOCKING_BYCALLER) {
    // The API doesn't let us report an error, so blow up.
    async_safe_fatal("Bad type (%d) passed to __fsetlocking", type);
  }

  _EXT(fp)->_caller_handles_locking = (type == FSETLOCKING_BYCALLER);
  return old_state;
}
