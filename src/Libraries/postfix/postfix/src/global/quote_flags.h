/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 8, 2022.
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
#include <vstring.h>

 /*
  * External interface.
  */
#define QUOTE_FLAG_8BITCLEAN	(1<<0)	/* be 8-bit clean */
#define QUOTE_FLAG_EXPOSE_AT	(1<<1)	/* @ is ordinary text */
#define QUOTE_FLAG_APPEND	(1<<2)	/* append, not overwrite */
#define QUOTE_FLAG_BARE_LOCALPART (1<<3)/* all localpart, no @domain */

#define QUOTE_FLAG_DEFAULT	QUOTE_FLAG_8BITCLEAN

extern int quote_flags_from_string(const char *);
extern const char *quote_flags_to_string(VSTRING *, int);

/* LICENSE
/* .ad
/* .fi
/*	The Secure Mailer license must be distributed with this software.
/* AUTHOR(S)
/*	Wietse Venema
/*	IBM T.J. Watson Research
/*	P.O. Box 704
/*	Yorktown Heights, NY 10598, USA
/*--*/
