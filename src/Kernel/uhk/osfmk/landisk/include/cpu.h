/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 8, 2022.
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
/*	$NetBSD: cpu.h,v 1.1 2006/09/01 21:26:18 uwe Exp $	*/

#ifndef	_MACHINE_CPU_H_
#define	_MACHINE_CPU_H_

void machine_reset(void);

#include <sh/cpu.h>

/*
 * CTL_MACHDEP definitions.
 */
#define	CPU_CONSDEV		1	/* dev_t: console terminal device */
		/*		2	   formerly: keyboard reset */
#define	CPU_LED_BLINK		3	/* blink leds */
#define	CPU_MAXID		4	/* number of valid machdep ids */

#define	CTL_MACHDEP_NAMES {						\
	{ 0, 0 },							\
	{ "console_device",	CTLTYPE_STRUCT },			\
	{ 0, 0 },							\
	{ "led_blink",		CTLTYPE_INT }				\
}

#endif	/* _MACHINE_CPU_H_ */
