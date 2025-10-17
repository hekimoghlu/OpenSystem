/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 27, 2023.
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
 * Indirect driver for console
 *
 * The purpose of this driver is to provide a device node indirection for
 * the console device, which can be any tty class device.  It does this by
 * externalizing a global pointer "constty", which is then pointed at the
 * console tty device.
 *
 * The default for this pointer is uninitialized; when it is NULL, we fall
 * back to the "km" device, which is a tty BSD wrapper device for the
 * Platform Expert console device.  When it is non-NULL, we call through
 * to the tty device device instead.
 *
 * The registration for this device node is static, and the devfs init
 * code does not externalize a named device for it, to avoid software
 * seeing the device and trying to open it.
 *
 * The upshot of this is that the console driver should not be set as your
 * controlling tty, since you will get a reference to a device which does
 * not have an actual device node in /dev, so its name cannot be looked up.
 */
#include <machine/cons.h>
#include <sys/param.h>
#include <sys/systm.h>
#include <sys/conf.h>
#include <sys/ioctl.h>
#include <sys/tty.h>
#include <sys/proc.h>
#include <sys/uio.h>

struct tty      *_constty;               /* current console device */
static LCK_GRP_DECLARE(constty_lock_grp, "constty");
static LCK_MTX_DECLARE(_constty_lock, &constty_lock_grp);

struct tty *
copy_constty(void)
{
	struct tty *result = NULL;
	lck_mtx_lock(&_constty_lock);
	if (_constty != NULL) {
		ttyhold(_constty);
		result = _constty;
	}
	lck_mtx_unlock(&_constty_lock);
	return result;
}

struct tty *
set_constty(struct tty *new_tty)
{
	struct tty *old_tty = NULL;
	lck_mtx_lock(&_constty_lock);
	old_tty = _constty;
	_constty = new_tty;
	if (_constty) {
		ttyhold(_constty);
	}
	lck_mtx_unlock(&_constty_lock);

	return old_tty;
}

/*
 * The km driver supplied the default console device for the systems
 * (usually a raw frame buffer driver, but potentially a serial driver).
 */
extern struct tty *km_tty[1];

/*
 * cdevsw[] entries for the console device driver
 */
int cnopen(__unused dev_t dev, int flag, int devtype, proc_t pp);
int cnclose(__unused dev_t dev, int flag, int mode, proc_t pp);
int cnread(__unused dev_t dev, struct uio *uio, int ioflag);
int cnwrite(__unused dev_t dev, struct uio *uio, int ioflag);
int cnioctl(__unused dev_t dev, u_long cmd, caddr_t addr, int flg, proc_t p);
int cnselect(__unused dev_t dev, int flag, void * wql, proc_t p);

int
cnopen(__unused dev_t dev, int flag, int devtype, struct proc *pp)
{
	int error;
	struct tty *constty = copy_constty();
	if (constty) {
		dev = constty->t_dev;
	} else {
		dev = km_tty[0]->t_dev;
	}
	error = (*cdevsw[major(dev)].d_open)(dev, flag, devtype, pp);
	if (constty != NULL) {
		ttyfree(constty);
	}
	return error;
}


int
cnclose(__unused dev_t dev, int flag, int mode, struct proc *pp)
{
	int error;
	struct tty *constty = copy_constty();
	if (constty) {
		dev = constty->t_dev;
	} else {
		dev = km_tty[0]->t_dev;
	}
	error = (*cdevsw[major(dev)].d_close)(dev, flag, mode, pp);
	if (constty != NULL) {
		ttyfree(constty);
	}
	return error;
}


int
cnread(__unused dev_t dev, struct uio *uio, int ioflag)
{
	int error;
	struct tty *constty = copy_constty();
	if (constty) {
		dev = constty->t_dev;
	} else {
		dev = km_tty[0]->t_dev;
	}
	error = (*cdevsw[major(dev)].d_read)(dev, uio, ioflag);
	if (constty != NULL) {
		ttyfree(constty);
	}
	return error;
}


int
cnwrite(__unused dev_t dev, struct uio *uio, int ioflag)
{
	int error;
	struct tty *constty = copy_constty();
	if (constty) {
		dev = constty->t_dev;
	} else {
		dev = km_tty[0]->t_dev;
	}
	error = (*cdevsw[major(dev)].d_write)(dev, uio, ioflag);
	if (constty != NULL) {
		ttyfree(constty);
	}
	return error;
}


int
cnioctl(__unused dev_t dev, u_long cmd, caddr_t addr, int flag, struct proc *p)
{
	int error;
	struct tty *constty = copy_constty();
	if (constty) {
		dev = constty->t_dev;
	} else {
		dev = km_tty[0]->t_dev;
	}
#if 0
	/*
	 * Superuser can always use this to wrest control of console
	 * output from the "virtual" console.
	 *
	 * XXX Unfortunately, this code doesn't do what the author thougt
	 * XXX it did; use of the console device, a TIOCCONS would always
	 * XXX disassociate the console from a virtual terminal and send
	 * XXX it back to the fake tty.
	 */
	if ((unsigned) cmd == TIOCCONS && constty) {
		int error = proc_suser(p);
		if (!error) {
			constty = NULL;
		}
		return error;
	}
#endif  /* 0 */

	error = (*cdevsw[major(dev)].d_ioctl)(dev, cmd, addr, flag, p);
	if (constty != NULL) {
		ttyfree(constty);
	}
	return error;
}


int
cnselect(__unused dev_t dev, int flag, void *wql, struct proc *p)
{
	int error;
	struct tty *constty = copy_constty();
	if (constty) {
		dev = constty->t_dev;
	} else {
		dev = km_tty[0]->t_dev;
	}
	error = (*cdevsw[major(dev)].d_select)(dev, flag, wql, p);
	if (constty != NULL) {
		ttyfree(constty);
	}
	return error;
}
