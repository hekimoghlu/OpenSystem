/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 2, 2024.
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
#define BYTESIZ		8

#ifdef SVID
#include <unistd.h>
#endif

#ifdef BSD42
#define SEEK_SET	L_SET
#define	memset(s,c,n)	bzero(s, n)		/* only when c is zero */
#define	memcpy(s1,s2,n)	bcopy(s2, s1, n)
#define	memcmp(s1,s2,n)	bcmp(s1,s2,n)
#endif

/*
 * important tuning parms (hah)
 */

#define SEEDUPS			/* always detect duplicates */
#define BADMESS			/* generate a message for worst case:
				   cannot make room after SPLTMAX splits */
/*
 * misc
 */
#ifdef DEBUG
#define debug(x)	printf x
#else
#define debug(x)
#endif
