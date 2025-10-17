/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 21, 2023.
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
#define	settimer(x)	clocks.x = clocks.system++

#define	NETADD(c)	{ *netoring.supply = c; ring_supplied(&netoring, 1); }
#define	NET2ADD(c1,c2)	{ NETADD(c1); NETADD(c2); }
#define	NETBYTES()	(ring_full_count(&netoring))
#define	NETROOM()	(ring_empty_count(&netoring))

#define	TTYADD(c)	if (!(SYNCHing||flushout)) { \
				*ttyoring.supply = c; \
				ring_supplied(&ttyoring, 1); \
			}
#define	TTYBYTES()	(ring_full_count(&ttyoring))
#define	TTYROOM()	(ring_empty_count(&ttyoring))

/*	Various modes */
#define	MODE_LOCAL_CHARS(m)	((m)&(MODE_EDIT|MODE_TRAPSIG))
#define	MODE_LOCAL_ECHO(m)	((m)&MODE_ECHO)
#define	MODE_COMMAND_LINE(m)	((m)==-1)

#define	CONTROL(x)	((x)&0x1f)		/* CTRL(x) is not portable */


/* XXX extra mode bits, these should be synced with <arpa/telnet.h> */

#define MODE_OUT8	0x8000 /* binary mode sans -opost */
