/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 23, 2025.
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
/* $Id: stdio.h,v 1.13 2007/06/19 23:47:18 tbox Exp $ */

#ifndef ISC_STDIO_H
#define ISC_STDIO_H 1

/*! \file isc/stdio.h */

/*%
 * These functions are wrappers around the corresponding stdio functions.
 *
 * They return a detailed error code in the form of an an isc_result_t.  ANSI C
 * does not guarantee that stdio functions set errno, hence these functions
 * must use platform dependent methods (e.g., the POSIX errno) to construct the
 * error code.
 */

#include <stdio.h>

#include <isc/lang.h>
#include <isc/result.h>

ISC_LANG_BEGINDECLS

/*% Open */
isc_result_t
isc_stdio_open(const char *filename, const char *mode, FILE **fp);

/*% Close */
isc_result_t
isc_stdio_close(FILE *f);

/*% Seek */
isc_result_t
isc_stdio_seek(FILE *f, off_t offset, int whence);

/*% Tell */
isc_result_t
isc_stdio_tell(FILE *f, off_t *offsetp);

/*% Read */
isc_result_t
isc_stdio_read(void *ptr, size_t size, size_t nmemb, FILE *f,
	       size_t *nret);

/*% Write */
isc_result_t
isc_stdio_write(const void *ptr, size_t size, size_t nmemb, FILE *f,
		size_t *nret);

/*% Flush */
isc_result_t
isc_stdio_flush(FILE *f);

isc_result_t
isc_stdio_sync(FILE *f);
/*%<
 * Invoke fsync() on the file descriptor underlying an stdio stream, or an
 * equivalent system-dependent operation.  Note that this function has no
 * direct counterpart in the stdio library.
 */

ISC_LANG_ENDDECLS

#endif /* ISC_STDIO_H */
