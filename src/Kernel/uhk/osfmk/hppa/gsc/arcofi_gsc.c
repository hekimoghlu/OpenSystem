/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 2, 2021.
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
 * Copyright (c) 2011 Miodrag Vallat.
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

#include <machine/autoconf.h>
#include <machine/bus.h>
#include <machine/cpu.h>
#include <machine/intr.h>
#include <machine/iomod.h>

#include <sys/audioio.h>
#include <dev/audio_if.h>
#include <dev/ic/arcofivar.h>

#include <hppa/dev/cpudevs.h>
#include <hppa/gsc/gscbusvar.h>

int	arcofi_gsc_match(struct device *, void *, void *);
void	arcofi_gsc_attach(struct device *, struct device *, void *);

const struct cfattach arcofi_gsc_ca = {
	sizeof(struct arcofi_softc),
	arcofi_gsc_match,
	arcofi_gsc_attach
};

int
arcofi_gsc_match(struct device *parent, void *match, void *vaa)
{
	struct gsc_attach_args *ga = vaa;

	if (ga->ga_type.iodc_type == HPPA_TYPE_FIO &&
	    (ga->ga_type.iodc_sv_model == HPPA_FIO_A1 ||
	     ga->ga_type.iodc_sv_model == HPPA_FIO_A1NB))
		return 1;

	return 0;
}

void
arcofi_gsc_attach(struct device *parent, struct device *self, void *vaa)
{
	struct arcofi_softc *sc = (struct arcofi_softc *)self;
	struct gsc_attach_args *ga = vaa;
	unsigned int u;

	for (u = 0; u < ARCOFI_NREGS; u++)
		sc->sc_reg[u] = (u << 2) | 0x01;

	sc->sc_iot = ga->ga_iot;
	if (bus_space_map(sc->sc_iot, ga->ga_hpa, ARCOFI_NREGS << 2, 0,
	    &sc->sc_ioh) != 0) {
		printf(": can't map registers\n");
		return;
	}

	/* XXX no generic IPL_SOFT level available */
	sc->sc_sih = softintr_establish(IPL_SOFTTTY, &arcofi_swintr, sc);
	if (sc->sc_sih == NULL) {
		printf(": can't register soft interrupt\n");
		return;
	}
	gsc_intr_establish((struct gsc_softc *)parent, ga->ga_irq,
	    IPL_AUDIO, arcofi_hwintr, sc, sc->sc_dev.dv_xname);

	printf("\n");

	arcofi_attach(sc, "gsc");
}

