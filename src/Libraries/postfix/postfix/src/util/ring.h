/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 16, 2024.
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

#ifndef _RING_H_INCLUDED_
#define _RING_H_INCLUDED_

/*++
/* NAME
/*	ring 3h
/* SUMMARY
/*	circular list management
/* SYNOPSIS
/*	#include <ring.h>
/* DESCRIPTION
/* .nf

 /*
  * External interface.
  */
typedef struct RING RING;

struct RING {
    RING   *succ;			/* successor */
    RING   *pred;			/* predecessor */
};

extern void ring_init(RING *);
extern void ring_prepend(RING *, RING *);
extern void ring_append(RING *, RING *);
extern void ring_detach(RING *);

#define ring_succ(c) ((c)->succ)
#define ring_pred(c) ((c)->pred)

#define RING_FOREACH(entry, head) \
    for (entry = ring_succ(head); entry != (head); entry = ring_succ(entry))

 /*
  * Typically, an application will embed a RING structure into a larger
  * structure that also contains application-specific members. This approach
  * gives us the best of both worlds. The application can still use the
  * generic RING primitives for manipulating RING structures. The macro below
  * transforms a pointer from RING structure to the structure that contains
  * it.
  */
#define RING_TO_APPL(ring_ptr,app_type,ring_member) \
    ((app_type *) (((char *) (ring_ptr)) - offsetof(app_type,ring_member)))

/* LICENSE
/* .ad
/* .fi
/*	The Secure Mailer license must be distributed with this software.
/* AUTHOR(S)
/*	Wietse Venema
/*	IBM T.J. Watson Research
/*	P.O. Box 704
/*	Yorktown Heights, NY 10598, USA
/* LAST MODIFICATION
/*	Tue Jan 28 16:50:20 EST 1997
/*--*/

#endif
