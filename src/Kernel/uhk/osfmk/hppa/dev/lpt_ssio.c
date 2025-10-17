/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 29, 2023.
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
 * Copyright (c) 2007 Mark Kettenis
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

#include <machine/bus.h>

#include <dev/ic/lptreg.h>
#include <dev/ic/lptvar.h>

#include <hppa/dev/ssiovar.h>

int lpt_ssio_match(struct device *, void *, void *);
void lpt_ssio_attach(struct device *, struct device *, void *);

const struct cfattach lpt_ssio_ca = {
	sizeof(struct lpt_softc), lpt_ssio_match, lpt_ssio_attach
};

int
lpt_ssio_match(struct device *parent, void *match, void *aux)
{
	struct cfdata *cf = match;
	struct ssio_attach_args *saa = aux;

	if (strcmp(saa->saa_name, "lpt") != 0)
		return (0);

	/* Check locators. */
	if (cf->ssiocf_irq != SSIO_UNK_IRQ && cf->ssiocf_irq != saa->saa_irq)
		return (0);

	return (1);
}

void
lpt_ssio_attach(struct device *parent, struct device *self, void *aux)
{
	struct lpt_softc *sc = (void *)self;
	struct ssio_attach_args *saa = aux;

	sc->sc_iot = saa->saa_iot;
	if (bus_space_map(sc->sc_iot, saa->saa_iobase, LPT_NPORTS,
	    0, &sc->sc_ioh)) {
		printf(": cannot map io space\n");
		return;
	}

	lpt_attach_common(sc);

	sc->sc_ih = ssio_intr_establish(IPL_TTY, saa->saa_irq,
	    lptintr, sc, sc->sc_dev.dv_xname);
}

