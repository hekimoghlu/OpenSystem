/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 26, 2022.
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

/* $OrigId: md5.h,v 1.2 2001/03/26 08:57:14 matz Exp $ */
/* $RoughId: md5.h,v 1.3 2002/02/24 08:14:31 knu Exp $ */
/* $Id: md5.h 52694 2015-11-21 04:35:57Z naruse $ */

#ifndef MD5_INCLUDED
#  define MD5_INCLUDED

#include "../defs.h"

/*
 * This code has some adaptations for the Ghostscript environment, but it
 * will compile and run correctly in any environment with 8-bit chars and
 * 32-bit ints.  Specifically, it assumes that if the following are
 * defined, they have the same meaning as in Ghostscript: P1, P2, P3.
 */

/* Define the state of the MD5 Algorithm. */
typedef struct md5_state_s {
    uint32_t count[2];	/* message length in bits, lsw first */
    uint32_t state[4];	/* digest buffer */
    uint8_t buffer[64];	/* accumulate block */
} MD5_CTX;

#ifdef RUBY
/* avoid name clash */
#define MD5_Init	rb_Digest_MD5_Init
#define MD5_Update	rb_Digest_MD5_Update
#define MD5_Finish	rb_Digest_MD5_Finish
#endif

int	MD5_Init _((MD5_CTX *pms));
void	MD5_Update _((MD5_CTX *pms, const uint8_t *data, size_t nbytes));
int	MD5_Finish _((MD5_CTX *pms, uint8_t *digest));

#define MD5_BLOCK_LENGTH		64
#define MD5_DIGEST_LENGTH		16
#define MD5_DIGEST_STRING_LENGTH	(MD5_DIGEST_LENGTH * 2 + 1)

#endif /* MD5_INCLUDED */
