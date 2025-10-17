/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 15, 2021.
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
 * Copyright (c) 2007 Ted Unangst <tedu@openbsd.org>
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
#include <machine/conf.h>
#include <sys/event.h>


int (*acpiapm_open)(dev_t, int, int, struct proc *);
int (*acpiapm_close)(dev_t, int, int, struct proc *);
int (*acpiapm_ioctl)(dev_t, u_long, caddr_t, int, struct proc *);
int (*acpiapm_kqfilter)(dev_t, struct knote *);

int
acpiapmopen(dev_t dev, int flag, int mode, struct proc *p)
{
	if (!acpiapm_open)
		return ENODEV;
	return acpiapm_open(dev, flag, mode, p);
}

int
acpiapmclose(dev_t dev, int flag, int mode, struct proc *p)
{
	if (!acpiapm_close)
		return ENODEV;
	return acpiapm_close(dev, flag, mode, p);
}

int
acpiapmioctl(dev_t dev, u_long cmd, caddr_t data, int flag, struct proc *p)
{
	if (!acpiapm_ioctl)
		return ENODEV;
	return acpiapm_ioctl(dev, cmd, data, flag, p);
}

int
acpiapmkqfilter(dev_t dev, struct knote *kn)
{
	if (!acpiapm_kqfilter)
		return EOPNOTSUPP;
	return acpiapm_kqfilter(dev, kn);
}

