/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 7, 2024.
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
/* $Id: keyboard.c,v 1.7 2007/06/19 23:47:19 tbox Exp $ */

#include <config.h>

#include <sys/types.h>

#include <windows.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>

#include <io.h>

#include <isc/keyboard.h>
#include <isc/util.h>

isc_result_t
isc_keyboard_open(isc_keyboard_t *keyboard) {
	int fd;

	REQUIRE(keyboard != NULL);

	fd = _fileno(stdin);
	if (fd < 0)
		return (ISC_R_IOERROR);

	keyboard->fd = fd;

	keyboard->result = ISC_R_SUCCESS;

	return (ISC_R_SUCCESS);
}

isc_result_t
isc_keyboard_close(isc_keyboard_t *keyboard, unsigned int sleeptime) {
	REQUIRE(keyboard != NULL);

	if (sleeptime > 0 && keyboard->result != ISC_R_CANCELED)
		(void)Sleep(sleeptime*1000);

	keyboard->fd = -1;

	return (ISC_R_SUCCESS);
}

isc_result_t
isc_keyboard_getchar(isc_keyboard_t *keyboard, unsigned char *cp) {
	ssize_t cc;
	unsigned char c;

	REQUIRE(keyboard != NULL);
	REQUIRE(cp != NULL);

	cc = read(keyboard->fd, &c, 1);
	if (cc < 0) {
		keyboard->result = ISC_R_IOERROR;
		return (keyboard->result);
	}

	*cp = c;

	return (ISC_R_SUCCESS);
}

isc_boolean_t
isc_keyboard_canceled(isc_keyboard_t *keyboard) {
	return (ISC_TF(keyboard->result == ISC_R_CANCELED));
}

