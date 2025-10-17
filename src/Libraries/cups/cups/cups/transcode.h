/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 15, 2024.
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
#ifndef _CUPS_TRANSCODE_H_
#  define _CUPS_TRANSCODE_H_

/*
 * Include necessary headers...
 */

#  include "language.h"

#  ifdef __cplusplus
extern "C" {
#  endif /* __cplusplus */


/*
 * Constants...
 */

#  define CUPS_MAX_USTRING	8192	/* Max size of Unicode string */


/*
 * Types...
 */

typedef unsigned char  cups_utf8_t;	/* UTF-8 Unicode/ISO-10646 unit */
typedef unsigned long  cups_utf32_t;	/* UTF-32 Unicode/ISO-10646 unit */
typedef unsigned short cups_ucs2_t;	/* UCS-2 Unicode/ISO-10646 unit */
typedef unsigned long  cups_ucs4_t;	/* UCS-4 Unicode/ISO-10646 unit */
typedef unsigned char  cups_sbcs_t;	/* SBCS Legacy 8-bit unit */
typedef unsigned short cups_dbcs_t;	/* DBCS Legacy 16-bit unit */
typedef unsigned long  cups_vbcs_t;	/* VBCS Legacy 32-bit unit */
					/* EUC uses 8, 16, 24, 32-bit */


/*
 * Prototypes...
 */

extern int	cupsCharsetToUTF8(cups_utf8_t *dest,
				  const char *src,
				  const int maxout,
				  const cups_encoding_t encoding) _CUPS_API_1_2;
extern int	cupsUTF8ToCharset(char *dest,
				  const cups_utf8_t *src,
				  const int maxout,
				  const cups_encoding_t encoding) _CUPS_API_1_2;
extern int	cupsUTF8ToUTF32(cups_utf32_t *dest,
				const cups_utf8_t *src,
				const int maxout) _CUPS_API_1_2;
extern int	cupsUTF32ToUTF8(cups_utf8_t *dest,
				const cups_utf32_t *src,
				const int maxout) _CUPS_API_1_2;

#  ifdef __cplusplus
}
#  endif /* __cplusplus */

#endif /* !_CUPS_TRANSCODE_H_ */
