/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 5, 2022.
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
 * Definitions for hashing page file format.
 */

/*
 * routines dealing with a data page
 *
 * page format:
 *	+------------------------------+
 * p	| n | keyoff | datoff | keyoff |
 * 	+------------+--------+--------+
 *	| datoff | free  |  ptr  | --> |
 *	+--------+---------------------+
 *	|	 F R E E A R E A       |
 *	+--------------+---------------+
 *	|  <---- - - - | data	       |
 *	+--------+-----+----+----------+
 *	|  key   | data     | key      |
 *	+--------+----------+----------+
 *
 * Pointer to the free space is always:  p[p[0] + 2]
 * Amount of free space on the page is:  p[p[0] + 1]
 */

/*
 * How many bytes required for this pair?
 *	2 shorts in the table at the top of the page + room for the
 *	key and room for the data
 *
 * We prohibit entering a pair on a page unless there is also room to append
 * an overflow page. The reason for this it that you can get in a situation
 * where a single key/data pair fits on a page, but you can't append an
 * overflow page and later you'd have to split the key/data and handle like
 * a big pair.
 * You might as well do this up front.
 */

#define	PAIRSIZE(K,D)	(2*sizeof(u_int16_t) + (K)->size + (D)->size)
#define BIGOVERHEAD	(4*sizeof(u_int16_t))
#define KEYSIZE(K)	(4*sizeof(u_int16_t) + (K)->size);
#define OVFLSIZE	(2*sizeof(u_int16_t))
#define FREESPACE(P)	((P)[(P)[0]+1])
#define	OFFSET(P)	((P)[(P)[0]+2])
#define PAIRFITS(P,K,D) \
	(((P)[2] >= REAL_KEY) && \
	    (PAIRSIZE((K),(D)) + OVFLSIZE) <= FREESPACE((P)))
#define PAGE_META(N)	(((N)+3) * sizeof(u_int16_t))

typedef struct {
	BUFHEAD *newp;
	BUFHEAD *oldp;
	BUFHEAD *nextp;
	u_int16_t next_addr;
}       SPLIT_RETURN;
