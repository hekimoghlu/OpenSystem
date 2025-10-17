/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 4, 2025.
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
 * Telnet server defines
 */

#ifndef __DEFS_H__
#define __DEFS_H__

#ifndef	BSD
# define	BSD 43
#endif

#if defined(PRINTOPTIONS) && defined(DIAGNOSTICS)
#define TELOPTS
#define TELCMDS
#define	SLC_NAMES
#endif

#if	!defined(TIOCSCTTY) && defined(TCSETCTTY)
# define	TIOCSCTTY TCSETCTTY
#endif

#ifndef TIOCPKT_FLUSHWRITE
#define TIOCPKT_FLUSHWRITE      0x02
#endif

#ifndef TIOCPKT_NOSTOP
#define TIOCPKT_NOSTOP  0x10
#endif

#ifndef TIOCPKT_DOSTOP
#define TIOCPKT_DOSTOP  0x20
#endif

/*
 * I/O data buffers defines
 */
#define	NETSLOP	64
#ifdef _CRAY
#undef BUFSIZ
#define BUFSIZ  2048
#endif

#define	NIACCUM(c)	{   *netip++ = c; \
			    ncc++; \
			}

/* clock manipulations */
#define	settimer(x)	(clocks.x = ++clocks.system)
#define	sequenceIs(x,y)	(clocks.x < clocks.y)

/*
 * Structures of information for each special character function.
 */
typedef struct {
	unsigned char	flag;		/* the flags for this function */
	cc_t		val;		/* the value of the special character */
} slcent, *Slcent;

typedef struct {
	slcent		defset;		/* the default settings */
	slcent		current;	/* the current settings */
	cc_t		*sptr;		/* a pointer to the char in */
					/* system data structures */
} slcfun, *Slcfun;

#ifdef DIAGNOSTICS
/*
 * Diagnostics capabilities
 */
#define	TD_REPORT	0x01	/* Report operations to client */
#define TD_EXERCISE	0x02	/* Exercise client's implementation */
#define TD_NETDATA	0x04	/* Display received data stream */
#define TD_PTYDATA	0x08	/* Display data passed to pty */
#define	TD_OPTIONS	0x10	/* Report just telnet options */
#endif /* DIAGNOSTICS */

/*
 * We keep track of each side of the option negotiation.
 */

#define	MY_STATE_WILL		0x01
#define	MY_WANT_STATE_WILL	0x02
#define	MY_STATE_DO		0x04
#define	MY_WANT_STATE_DO	0x08

/*
 * Macros to check the current state of things
 */

#define	my_state_is_do(opt)		(options[opt]&MY_STATE_DO)
#define	my_state_is_will(opt)		(options[opt]&MY_STATE_WILL)
#define my_want_state_is_do(opt)	(options[opt]&MY_WANT_STATE_DO)
#define my_want_state_is_will(opt)	(options[opt]&MY_WANT_STATE_WILL)

#define	my_state_is_dont(opt)		(!my_state_is_do(opt))
#define	my_state_is_wont(opt)		(!my_state_is_will(opt))
#define my_want_state_is_dont(opt)	(!my_want_state_is_do(opt))
#define my_want_state_is_wont(opt)	(!my_want_state_is_will(opt))

#define	set_my_state_do(opt)		(options[opt] |= MY_STATE_DO)
#define	set_my_state_will(opt)		(options[opt] |= MY_STATE_WILL)
#define	set_my_want_state_do(opt)	(options[opt] |= MY_WANT_STATE_DO)
#define	set_my_want_state_will(opt)	(options[opt] |= MY_WANT_STATE_WILL)

#define	set_my_state_dont(opt)		(options[opt] &= ~MY_STATE_DO)
#define	set_my_state_wont(opt)		(options[opt] &= ~MY_STATE_WILL)
#define	set_my_want_state_dont(opt)	(options[opt] &= ~MY_WANT_STATE_DO)
#define	set_my_want_state_wont(opt)	(options[opt] &= ~MY_WANT_STATE_WILL)

/*
 * Tricky code here.  What we want to know is if the MY_STATE_WILL
 * and MY_WANT_STATE_WILL bits have the same value.  Since the two
 * bits are adjacent, a little arithmatic will show that by adding
 * in the lower bit, the upper bit will be set if the two bits were
 * different, and clear if they were the same.
 */
#define my_will_wont_is_changing(opt) \
			((options[opt]+MY_STATE_WILL) & MY_WANT_STATE_WILL)

#define my_do_dont_is_changing(opt) \
			((options[opt]+MY_STATE_DO) & MY_WANT_STATE_DO)

/*
 * Make everything symmetrical
 */

#define	HIS_STATE_WILL			MY_STATE_DO
#define	HIS_WANT_STATE_WILL		MY_WANT_STATE_DO
#define HIS_STATE_DO			MY_STATE_WILL
#define HIS_WANT_STATE_DO		MY_WANT_STATE_WILL

#define	his_state_is_do			my_state_is_will
#define	his_state_is_will		my_state_is_do
#define his_want_state_is_do		my_want_state_is_will
#define his_want_state_is_will		my_want_state_is_do

#define	his_state_is_dont		my_state_is_wont
#define	his_state_is_wont		my_state_is_dont
#define his_want_state_is_dont		my_want_state_is_wont
#define his_want_state_is_wont		my_want_state_is_dont

#define	set_his_state_do		set_my_state_will
#define	set_his_state_will		set_my_state_do
#define	set_his_want_state_do		set_my_want_state_will
#define	set_his_want_state_will		set_my_want_state_do

#define	set_his_state_dont		set_my_state_wont
#define	set_his_state_wont		set_my_state_dont
#define	set_his_want_state_dont		set_my_want_state_wont
#define	set_his_want_state_wont		set_my_want_state_dont

#define his_will_wont_is_changing	my_do_dont_is_changing
#define his_do_dont_is_changing		my_will_wont_is_changing

#endif /* __DEFS_H__ */
