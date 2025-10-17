/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 28, 2025.
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
#include <time.h>

 /*
  * Utility library.
  */
#include <vstring.h>

 /*
  * Global library.
  */
#include <maps.h>

 /*
  * External interface.
  */
typedef struct TRANSPORT_INFO {
    MAPS   *transport_path;
    VSTRING *wildcard_channel;
    VSTRING *wildcard_nexthop;
    int     wildcard_errno;
    time_t  expire;
} TRANSPORT_INFO;

extern TRANSPORT_INFO *transport_pre_init(const char *, const char *);
extern void transport_post_init(TRANSPORT_INFO *);
extern int transport_lookup(TRANSPORT_INFO *, const char *, const char *, VSTRING *, VSTRING *);
extern void transport_free(TRANSPORT_INFO *);

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
