/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 8, 2025.
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

   TODO: Add some documentation here.

   the SIGPIPE signal should be ignored (is ignored in this code)

   This library is not thread safe. You cannot share TFILE objects between
   threads and expect to be able to read and write from them in different
   threads. All the state is in the TFILE object so calls to this library on
   different objects can be done in parallel.

*/

#ifndef _TIO_H
#define _TIO_H

#include <sys/time.h>
#include <sys/types.h>

#include "attrs.h"

/* This is a generic file handle used for reading and writing
   (something like FILE from stdio.h). */
typedef struct tio_fileinfo TFILE;

/* Open a new TFILE based on the file descriptor. The timeout is set for any
   operation. The timeout value is copied so may be dereferenced after the
   call. */
TFILE *tio_fdopen(int fd,struct timeval *readtimeout,struct timeval *writetimeout,
                  size_t initreadsize,size_t maxreadsize,
                  size_t initwritesize,size_t maxwritesize)
  LIKE_MALLOC MUST_USE;

/* Read the specified number of bytes from the stream. */
int tio_read(TFILE *fp,void *buf,size_t count);

/* Read and discard the specified number of bytes from the stream. */
int tio_skip(TFILE *fp,size_t count);

/* Write the specified buffer to the stream. */
int tio_write(TFILE *fp,const void *buf,size_t count);

/* Write out all buffered data to the stream. */
int tio_flush(TFILE *fp);

/* Flush the streams and closes the underlying file descriptor. */
int tio_close(TFILE *fp);

/* Store the current position in the stream so that we can jump back to it
   with the tio_reset() function. */
void tio_mark(TFILE *fp);

/* Rewinds the stream to the point set by tio_mark(). Note that this only
   resets the read stream and not the write stream. This function returns
   whether the reset was successful (this function may fail if the buffers
   were full). */
int tio_reset(TFILE *fp);

#endif /* _TIO_H */
