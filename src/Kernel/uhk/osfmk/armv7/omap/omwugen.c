/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 21, 2025.
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
 * Copyright (c) 2016 Mark Kettenis
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

#include <machine/fdt.h>

#include <dev/ofw/openfirm.h>

struct omwugen_softc {
	struct device	sc_dev;
	struct interrupt_controller sc_ic;
};

int	omwugen_match(struct device *, void *, void *);
void	omwugen_attach(struct device *, struct device *, void *);

const struct cfattach omwugen_ca = {
	sizeof(struct omwugen_softc), omwugen_match, omwugen_attach
};

struct cfdriver omwugen_cd = {
	NULL, "omwugen", DV_DULL
};

int
omwugen_match(struct device *parent, void *match, void *aux)
{
	struct fdt_attach_args *faa = aux;

	return OF_is_compatible(faa->fa_node, "ti,omap4-wugen-mpu");
}

void
omwugen_attach(struct device *parent, struct device *self, void *aux)
{
	struct fdt_attach_args *faa = aux;
	struct omwugen_softc *sc = (struct omwugen_softc *)self;

	sc->sc_ic.ic_node = faa->fa_node;
	sc->sc_ic.ic_cookie = &sc->sc_ic;
	sc->sc_ic.ic_establish = arm_intr_parent_establish_fdt;
	sc->sc_ic.ic_disestablish = arm_intr_parent_disestablish_fdt;
	arm_intr_register_fdt(&sc->sc_ic);

	printf("\n");
}

