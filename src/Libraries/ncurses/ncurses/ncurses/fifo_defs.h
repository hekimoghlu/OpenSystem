/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 7, 2024.
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
/****************************************************************************
 *  Author: Zeyd M. Ben-Halim <zmbenhal@netcom.com> 1992,1995               *
 *     and: Eric S. Raymond <esr@snark.thyrsus.com>                         *
 ****************************************************************************/

/*
 * Common macros for lib_getch.c, lib_ungetch.c
 *
 * $Id: fifo_defs.h,v 1.7 2012/08/04 15:59:17 tom Exp $
 */

#ifndef FIFO_DEFS_H
#define FIFO_DEFS_H 1

#define head	sp->_fifohead
#define tail	sp->_fifotail
/* peek points to next uninterpreted character */
#define peek	sp->_fifopeek

#define h_inc() { \
	    (head >= FIFO_SIZE-1) \
		? head = 0 \
		: head++; \
	    if (head == tail) \
		head = -1, tail = 0; \
	}
#define h_dec() { \
	    (head <= 0) \
		? head = FIFO_SIZE-1 \
		: head--; \
	    if (head == tail) \
		tail = -1; \
	}
#define t_inc() { \
	    (tail >= FIFO_SIZE-1) \
		? tail = 0 \
		: tail++; \
	    if (tail == head) \
		tail = -1; \
	    }
#define t_dec() { \
	    (tail <= 0) \
		? tail = FIFO_SIZE-1 \
		: tail--; \
	    if (head == tail) \
		fifo_clear(sp); \
	    }
#define p_inc() { \
	    (peek >= FIFO_SIZE-1) \
		? peek = 0 \
		: peek++; \
	    }

#define cooked_key_in_fifo()	((head >= 0) && (peek != head))
#define raw_key_in_fifo()	((head >= 0) && (peek != tail))

#undef HIDE_EINTR

#endif /* FIFO_DEFS_H */
