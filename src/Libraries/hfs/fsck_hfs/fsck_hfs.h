/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 7, 2023.
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
#include "lib_fsck_hfs.h"

#define     EEXIT           8   /* Standard error exit. */

char*       unrawname(char *name);
char*       rawname(char *name);
void        cleanup_fs_fd(void);
void		catch __P((int));
void        ckfini ();
void        pfatal(char *fmt, va_list ap);
void        pwarn(char *fmt, va_list ap);
void		logstring(void *, const char *) __printflike(2, 0);     // write to log file 
void		outstring(void *, const char *) __printflike(2, 0);     // write to standard out
void 		llog(const char *fmt, ...) __printflike(1, 2);          // write to log file
void 		olog(const char *fmt, ...) __printflike(1, 2);          // write to standard out
void        plog(const char *fmt, ...) __printflike(1, 2);          // printf replacement that writes to both log file and standard out
void        vplog(const char *fmt, va_list ap) __printflike(1, 0);  // vprintf replacement that writes to both log file and standard out
void        fplog(FILE *stream, const char *fmt, va_list ap) __printflike(2, 3);    // fprintf replacement that writes to both log file and standard out
#define printf  plog      // just in case someone tries to use printf/fprint
#define fprintf fplog

void		DumpData(const void *ptr, size_t sz, char *label);

