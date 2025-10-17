/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 30, 2022.
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
/* Version History:
   1.0  26 Nov 2004  First version
   2.0  25 Apr 2008  Complete redesign for recovery of interrupted operations
                     Interface changed slightly in that now path is a prefix
                     Compression now occurs as needed during gzlog_write()
                     gzlog_write() now always leaves the log file as valid gzip
   2.1   8 Jul 2012  Fix argument checks in gzlog_compress() and gzlog_write()
   2.2  14 Aug 2012  Clean up signed comparisons
 */

/*
   The gzlog object allows writing short messages to a gzipped log file,
   opening the log file locked for small bursts, and then closing it.  The log
   object works by appending stored (uncompressed) data to the gzip file until
   1 MB has been accumulated.  At that time, the stored data is compressed, and
   replaces the uncompressed data in the file.  The log file is truncated to
   its new size at that time.  After each write operation, the log file is a
   valid gzip file that can decompressed to recover what was written.

   The gzlog operations can be interupted at any point due to an application or
   system crash, and the log file will be recovered the next time the log is
   opened with gzlog_open().
 */

#ifndef GZLOG_H
#define GZLOG_H

/* gzlog object type */
typedef void gzlog;

/* Open a gzlog object, creating the log file if it does not exist.  Return
   NULL on error.  Note that gzlog_open() could take a while to complete if it
   has to wait to verify that a lock is stale (possibly for five minutes), or
   if there is significant contention with other instantiations of this object
   when locking the resource.  path is the prefix of the file names created by
   this object.  If path is "foo", then the log file will be "foo.gz", and
   other auxiliary files will be created and destroyed during the process:
   "foo.dict" for a compression dictionary, "foo.temp" for a temporary (next)
   dictionary, "foo.add" for data being added or compressed, "foo.lock" for the
   lock file, and "foo.repairs" to log recovery operations performed due to
   interrupted gzlog operations.  A gzlog_open() followed by a gzlog_close()
   will recover a previously interrupted operation, if any. */
gzlog *gzlog_open(char *path);

/* Write to a gzlog object.  Return zero on success, -1 if there is a file i/o
   error on any of the gzlog files (this should not happen if gzlog_open()
   succeeded, unless the device has run out of space or leftover auxiliary
   files have permissions or ownership that prevent their use), -2 if there is
   a memory allocation failure, or -3 if the log argument is invalid (e.g. if
   it was not created by gzlog_open()).  This function will write data to the
   file uncompressed, until 1 MB has been accumulated, at which time that data
   will be compressed.  The log file will be a valid gzip file upon successful
   return. */
int gzlog_write(gzlog *log, void *data, size_t len);

/* Force compression of any uncompressed data in the log.  This should be used
   sparingly, if at all.  The main application would be when a log file will
   not be appended to again.  If this is used to compress frequently while
   appending, it will both significantly increase the execution time and
   reduce the compression ratio.  The return codes are the same as for
   gzlog_write(). */
int gzlog_compress(gzlog *log);

/* Close a gzlog object.  Return zero on success, -3 if the log argument is
   invalid.  The log object is freed, and so cannot be referenced again. */
int gzlog_close(gzlog *log);

#endif
