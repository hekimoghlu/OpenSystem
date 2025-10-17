/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 22, 2024.
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
 * Copyright (c) 2005 Mark Kettenis
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

#include <sys/param.h>
#include <sys/systm.h>
#include <sys/device.h>

#define _I2C_PRIVATE
#include <dev/i2c/i2cvar.h>
#include <dev/ofw/openfirm.h>

#include <arch/macppc/dev/maci2cvar.h>

void
maciic_scan(struct device *self, struct i2cbus_attach_args *iba, void *aux)
{
	int iba_node = *(int *)aux;
	struct i2c_attach_args ia;
	char name[32];
	u_int32_t reg;
	int node;

	for (node = OF_child(iba_node); node; node = OF_peer(node)) {
		if (OF_getprop(node, "reg", &reg, sizeof reg) != sizeof reg &&
		    OF_getprop(node, "i2c-address", &reg, sizeof reg) != sizeof reg)
			continue;
		bzero(&ia, sizeof ia);
		ia.ia_tag = iba->iba_tag;
		ia.ia_addr = (reg >> 1);
		ia.ia_cookie = &node;
		bzero(name, sizeof name);
		if (OF_getprop(node, "compatible", &name,
		    sizeof name) && name[0])
			ia.ia_name = name;
		if (ia.ia_name == NULL && 
		    OF_getprop(node, "name", &name,
		    sizeof name) && name[0]) {
			if (strcmp(name, "cereal") == 0)
				continue;
			ia.ia_name = name;
		}
		/* XXX We should write a real driver for these instead
		   of reaching over from the sound driver that sits on
		   the i2s port.  For now hide them.  */
		if (strcmp(name, "deq") == 0 || strcmp(name, "tas3004") == 0)
			continue;
		if (ia.ia_name)
			config_found(self, &ia, iic_print);
	}
}

