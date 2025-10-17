/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 14, 2024.
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
#include <architecture/i386/pio.h>
#include <i386/panic_notify.h>
#include <kern/assert.h>
#include <pexpert/pexpert.h>
#include <stdint.h>

/*
 * An I/O port to issue a read from, in the event of a panic.
 * Useful for triggering logic analyzers.
 */
static uint16_t panic_io_port = 0;

/*
 * Similar to the panic_io_port, the pvpanic_io_port is used to notify
 * interested parties (in this case the host/hypervisor), that a panic
 * has occurred.
 * Where it differs from panic_io_port is that it is written and read
 * according to the pvpanic specification:
 * https://raw.githubusercontent.com/qemu/qemu/master/docs/specs/pvpanic.txt
 */
static uint16_t pvpanic_io_port = 0;

void
panic_notify_init(void)
{
	(void) PE_parse_boot_argn("panic_io_port", &panic_io_port, sizeof(panic_io_port));

	/*
	 * XXX
	 * Defer reading the notifcation bit until panic time. This maintains
	 * backwards compatibility with Apple's QEMU. Once backwards
	 * compatibilty is no longer needed the check should be performed here
	 * before setting pvpanic_io_port.
	 */
	(void) PE_parse_boot_argn("pvpanic_io_port", &pvpanic_io_port, sizeof(pvpanic_io_port));
}

void
panic_notify(void)
{
	if (panic_io_port != 0) {
		(void) inb(panic_io_port);
	}

	if (pvpanic_io_port != 0 &&
	    (inb(pvpanic_io_port) & PVPANIC_NOTIFICATION_BIT) != 0) {
		outb(pvpanic_io_port, PVPANIC_NOTIFICATION_BIT);
	}
}
