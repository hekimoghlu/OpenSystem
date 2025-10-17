/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 22, 2023.
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
  Independent implementation of MD5 (RFC 1321).

  This code implements the MD5 Algorithm defined in RFC 1321.
  It is derived directly from the text of the RFC and not from the
  reference implementation.

  The original and principal author of md5.h is L. Peter Deutsch
  <ghost@aladdin.com>.  Other authors are noted in the change history
  that follows (in reverse chronological order):

  1999-11-04 lpd Edited comments slightly for automatic TOC extraction.
  1999-10-18 lpd Fixed typo in header comment (ansi2knr rather than md5);
	added conditionalization for C++ compilation from Martin
	Purschke <purschke@bnl.gov>.
  1999-05-03 lpd Original version.
 */

#ifndef _CUPS_MD5_INTERNAL_H_
#  define _CUPS_MD5_INTERNAL_H_

#  include <cups/versioning.h>

/* Define the state of the MD5 Algorithm. */
typedef struct _cups_md5_state_s {
    unsigned int count[2];		/* message length in bits, lsw first */
    unsigned int abcd[4];		/* digest buffer */
    unsigned char buf[64];		/* accumulate block */
} _cups_md5_state_t;

#  ifdef __cplusplus
extern "C" {
#  endif /* __cplusplus */

/* Initialize the algorithm. */
void _cupsMD5Init(_cups_md5_state_t *pms) _CUPS_INTERNAL;

/* Append a string to the message. */
void _cupsMD5Append(_cups_md5_state_t *pms, const unsigned char *data, int nbytes) _CUPS_INTERNAL;

/* Finish the message and return the digest. */
void _cupsMD5Finish(_cups_md5_state_t *pms, unsigned char digest[16]) _CUPS_INTERNAL;

#  ifdef __cplusplus
}  /* end extern "C" */
#  endif /* __cplusplus */
#endif /* !_CUPS_MD5_INTERNAL_H_ */
