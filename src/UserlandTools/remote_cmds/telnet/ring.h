/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 8, 2023.
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
#if defined(P)
# undef P
#endif

#if defined(__STDC__) || defined(LINT_ARGS)
# define	P(x)	x
#else
# define	P(x)	()
#endif

/*
 * This defines a structure for a ring buffer.
 *
 * The circular buffer has two parts:
 *(((
 *	full:	[consume, supply)
 *	empty:	[supply, consume)
 *]]]
 *
 */
typedef struct {
    unsigned char	*consume,	/* where data comes out of */
			*supply,	/* where data comes in to */
			*bottom,	/* lowest address in buffer */
			*top,		/* highest address+1 in buffer */
			*mark;		/* marker (user defined) */
#ifdef	ENCRYPTION
    unsigned char	*clearto;	/* Data to this point is clear text */
    unsigned char	*encryyptedto;	/* Data is encrypted to here */
#endif	/* ENCRYPTION */
    int		size;		/* size in bytes of buffer */
    u_long	consumetime,	/* help us keep straight full, empty, etc. */
		supplytime;
} Ring;

/* Here are some functions and macros to deal with the ring buffer */

/* Initialization routine */
extern int
	ring_init(Ring *ring, unsigned char *buffer, int count);

/* Data movement routines */
extern void
	ring_supply_data(Ring *ring, unsigned char *buffer, int count);
#ifdef notdef
extern void
	ring_consume_data(Ring *ring, unsigned char *buffer, int count);
#endif

/* Buffer state transition routines */
extern void
	ring_supplied(Ring *ring, int count),
	ring_consumed(Ring *ring, int count);

/* Buffer state query routines */
extern int
	ring_at_mark(Ring *),
	ring_empty_count(Ring *ring),
	ring_empty_consecutive(Ring *ring),
	ring_full_count(Ring *ring),
	ring_full_consecutive(Ring *ring);

#ifdef	ENCRYPTION
extern void
	ring_encrypt(Ring *ring, void (*func)(unsigned char *, int)),
	ring_clearto(Ring *ring);
#endif	/* ENCRYPTION */

extern void
	ring_clear_mark(Ring *),
	ring_mark(Ring *);
