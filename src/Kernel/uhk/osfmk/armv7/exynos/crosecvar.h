/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 25, 2022.
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
 * Copyright (c) 2013 Patrick Wildt <patrick@blueri.se>
 *
 * Permission to use, copy, modify, and distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 * ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 * OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */

#ifndef CROSECVAR_H
#define CROSECVAR_H

#include <sys/timeout.h>
#include <sys/task.h>
#include <dev/i2c/i2cvar.h>
#include <armv7/exynos/ec_commands.h>

/* message sizes */
#define MSG_HEADER		0xec
#define MSG_HEADER_BYTES	3
#define MSG_TRAILER_BYTES	2
#define MSG_PROTO_BYTES		(MSG_HEADER_BYTES + MSG_TRAILER_BYTES)
#define MSG_BYTES		(EC_HOST_PARAM_SIZE + MSG_PROTO_BYTES)
#define MSG_BYTES_ALIGNED	((MSG_BYTES+8) & ~8)

#define min(a,b)	(((a)<(b))?(a):(b))

struct cros_ec_softc {
	struct device sc_dev;
	i2c_tag_t sc_tag;
	i2c_addr_t sc_addr;

	int cmd_version_is_supported;
	struct {
		int rows;
		int cols;
		int switches;
		uint8_t *state;

		/* wskbd bits */
		struct device *wskbddev;
		int rawkbd;
		int polling;

		/* polling */
		struct timeout timeout;
		struct taskq *taskq;
		struct task task;
	} keyboard;
	uint8_t in[MSG_BYTES_ALIGNED];
	uint8_t out[MSG_BYTES_ALIGNED];
};

int	cros_ec_check_version(struct cros_ec_softc *);
int	cros_ec_scan_keyboard(struct cros_ec_softc *, uint8_t *, int);
int	cros_ec_info(struct cros_ec_softc *, struct ec_response_cros_ec_info *);

int	cros_ec_init_keyboard(struct cros_ec_softc *);

#endif /* !CROSECVAR_H */
