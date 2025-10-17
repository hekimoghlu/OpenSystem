/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 23, 2022.
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
 * Copyright (C) 1998	Colin Wood
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *	This product includes software developed by Colin Wood.
 * 4. The name of the author may not be used to endorse or promote products
 *    derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <sys/param.h>
#include <sys/timeout.h>
#include <sys/kernel.h>
#include <sys/device.h>
#include <sys/fcntl.h>
#include <sys/proc.h>
#include <sys/signalvar.h>
#include <sys/systm.h>

#include <dev/wscons/wsconsio.h>
#include <dev/wscons/wskbdvar.h>

#include <dev/adb/adb.h>
#include <dev/adb/akbdvar.h>
#include <dev/adb/keyboard.h>

void	akbd_cngetc(void *, u_int *, int *);
void	akbd_cnpollc(void *, int);

struct wskbd_consops akbd_consops = {
	akbd_cngetc,
	akbd_cnpollc,
};

static int _akbd_is_console;

int
akbd_is_console(void)
{
	return (_akbd_is_console);
}

int
akbd_cnattach(void)
{
	_akbd_is_console = 1;
	wskbd_cnattach(&akbd_consops, NULL, &akbd_keymapdata);
	return 0;
}

void
akbd_cngetc(void *v, u_int *type, int *data)
{
	int key, press, val;
	int s;
	extern int adb_intr(void *);

	s = splhigh();

	adb_polledkey = -1;

	while (adb_polledkey == -1) {
		adb_intr(NULL); /* adb does not use the argument */
		DELAY(10000);				/* XXX */
	}

	splx(s);

	key = adb_polledkey;
	press = ADBK_PRESS(key);
	val = ADBK_KEYVAL(key);

	*data = val;
	*type = press ? WSCONS_EVENT_KEY_DOWN : WSCONS_EVENT_KEY_UP;
}

void
akbd_cnpollc(void *v, int on)
{
	adb_polling = on;
}

