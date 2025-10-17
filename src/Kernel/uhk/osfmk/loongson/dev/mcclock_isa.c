/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 9, 2022.
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
 * Copyright (c) 1995, 1996 Carnegie-Mellon University.
 * All rights reserved.
 *
 * Author: Chris G. Demetriou
 *
 * Permission to use, copy, modify and distribute this software and
 * its documentation is hereby granted, provided that both the copyright
 * notice and this permission notice appear in all copies of the
 * software, derivative works or modified versions, and any portions
 * thereof, and that both notices appear in supporting documentation.
 *
 * CARNEGIE MELLON ALLOWS FREE USE OF THIS SOFTWARE IN ITS "AS IS"
 * CONDITION.  CARNEGIE MELLON DISCLAIMS ANY LIABILITY OF ANY KIND
 * FOR ANY DAMAGES WHATSOEVER RESULTING FROM THE USE OF THIS SOFTWARE.
 *
 * Carnegie Mellon requests users of this software to return to
 *
 *  Software Distribution Coordinator  or  Software.Distribution@CS.CMU.EDU
 *  School of Computer Science
 *  Carnegie Mellon University
 *  Pittsburgh PA 15213-3890
 *
 * any improvements or extensions that they make and grant Carnegie the
 * rights to redistribute these changes.
 */

#include <sys/param.h>
#include <sys/kernel.h>
#include <sys/systm.h>
#include <sys/device.h>

#include <machine/bus.h>

#include <dev/clock_subr.h>
#include <dev/ic/mc146818reg.h>
#include <dev/isa/isavar.h>

#include <loongson/dev/mcclockvar.h>

struct mcclock_isa_softc {
	struct mcclock_softc	sc_mcclock;

	bus_space_tag_t		sc_iot;
	bus_space_handle_t	sc_ioh;
};

int	mcclock_isa_match(struct device *, void *, void *);
void	mcclock_isa_attach(struct device *, struct device *, void *);

const struct cfattach mcclock_isa_ca = {
	sizeof (struct mcclock_isa_softc), mcclock_isa_match,
	    mcclock_isa_attach, 
};

void	mcclock_isa_write(struct mcclock_softc *, u_int, u_int);
u_int	mcclock_isa_read(struct mcclock_softc *, u_int);

const struct mcclock_busfns mcclock_isa_busfns = {
	mcclock_isa_write, mcclock_isa_read,
};

int
mcclock_isa_match(struct device *parent, void *match, void *aux)
{
	struct isa_attach_args *ia = aux;
	bus_space_handle_t ioh;

        if ((ia->ia_iobase != IOBASEUNK && ia->ia_iobase != 0x70) ||
            /* (ia->ia_iosize != 0 && ia->ia_iosize != 0x2) || XXX isa.c */
            ia->ia_maddr != MADDRUNK || ia->ia_msize != 0 ||
            ia->ia_irq != IRQUNK || ia->ia_drq != DRQUNK)
                return (0);

	if (bus_space_map(ia->ia_iot, 0x70, 0x2, 0, &ioh))
		return (0);

	bus_space_unmap(ia->ia_iot, ioh, 0x2);

	ia->ia_iobase = 0x70;
	ia->ia_iosize = 0x2;

	return (1);
}

void
mcclock_isa_attach(struct device *parent, struct device *self, void *aux)
{
	struct isa_attach_args *ia = aux;
	struct mcclock_isa_softc *sc = (struct mcclock_isa_softc *)self;

	sc->sc_iot = ia->ia_iot;
	if (bus_space_map(sc->sc_iot, ia->ia_iobase, ia->ia_iosize, 0,
	    &sc->sc_ioh))
		panic("mcclock_isa_attach: couldn't map clock I/O space");

	mcclock_attach(&sc->sc_mcclock, &mcclock_isa_busfns);
}

void
mcclock_isa_write(struct mcclock_softc *mcsc, u_int reg, u_int datum)
{
	struct mcclock_isa_softc *sc = (struct mcclock_isa_softc *)mcsc;
	bus_space_tag_t iot = sc->sc_iot;
	bus_space_handle_t ioh = sc->sc_ioh;

	bus_space_write_1(iot, ioh, 0, reg);
	bus_space_write_1(iot, ioh, 1, datum);
}

u_int
mcclock_isa_read(struct mcclock_softc *mcsc, u_int reg)
{
	struct mcclock_isa_softc *sc = (struct mcclock_isa_softc *)mcsc;
	bus_space_tag_t iot = sc->sc_iot;
	bus_space_handle_t ioh = sc->sc_ioh;

	bus_space_write_1(iot, ioh, 0, reg);
	return bus_space_read_1(iot, ioh, 1);
}

