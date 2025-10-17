/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 3, 2023.
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
 * @OSF_COPYRIGHT@
 */
/*
 * Mach Operating System
 * Copyright (c) 1991,1990,1989 Carnegie Mellon University
 * All Rights Reserved.
 *
 * Permission to use, copy, modify and distribute this software and its
 * documentation is hereby granted, provided that both the copyright
 * notice and this permission notice appear in all copies of the
 * software, derivative works or modified versions, and any portions
 * thereof, and that both notices appear in supporting documentation.
 *
 * CARNEGIE MELLON ALLOWS FREE USE OF THIS SOFTWARE IN ITS
 * CONDITION.  CARNEGIE MELLON DISCLAIMS ANY LIABILITY OF ANY KIND FOR
 * ANY DAMAGES WHATSOEVER RESULTING FROM THE USE OF THIS SOFTWARE.
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

/*
 *	File:	error_codes.c
 *	Author:	Douglas Orr, Carnegie Mellon University
 *	Date:	Mar, 1988
 *
 *      Generic error code interface
 */

#include <TargetConditionals.h>
#include <mach/error.h>
#include "errorlib.h"
#if !TARGET_OS_DRIVERKIT
#include "err_libkern.sub"
// #include "err_iokit.sub" // FIXME: Re-enable once IOFireWireLib.h, IOUSBLib.h are found
#endif // !TARGET_OS_DRIVERKIT
#include "err_ipc.sub"
#include "err_kern.sub"
#include "err_mach_ipc.sub"
#include "err_server.sub"
#include "err_us.sub"

const struct error_system _mach_errors[err_max_system + 1] = {
	/* 0; err_kern */
	{
		.max_sub = errlib_count(err_os_sub),
		.bad_sub = "(operating system/?) unknown subsystem error",
		.subsystem = err_os_sub,
	},
	/* 1; err_us */
	{
		.max_sub = errlib_count(err_us_sub),
		.bad_sub = "(user space/?) unknown subsystem error",
		.subsystem = err_us_sub,
	},
	/* 2; err_server */
	{
		.max_sub = errlib_count(err_server_sub),
		.bad_sub = "(server/?) unknown subsystem error",
		.subsystem = err_server_sub,
	},
	/* 3 (& 3f); err_ipc */
	{
		.max_sub = errlib_count(err_ipc_sub),
		.bad_sub = "(ipc/?) unknown subsystem error",
		.subsystem = err_ipc_sub,
	},
	/* 4; err_mach_ipc */
	{
		.max_sub = errlib_count(err_mach_ipc_sub),
		.bad_sub = "(ipc/?) unknown subsystem error",
		.subsystem = err_mach_ipc_sub,
	},

	/* 0x05 */ errorlib_sub_null,
	/* 0x06 */ errorlib_sub_null, /* 0x07 */ errorlib_sub_null,
	/* 0x08 */ errorlib_sub_null, /* 0x09 */ errorlib_sub_null,
	/* 0x0a */ errorlib_sub_null, /* 0x0b */ errorlib_sub_null,
	/* 0x0c */ errorlib_sub_null, /* 0x0d */ errorlib_sub_null,
	/* 0x0e */ errorlib_sub_null, /* 0x0f */ errorlib_sub_null,

	/* 0x10 */ errorlib_sub_null, /* 0x11 */ errorlib_sub_null,
	/* 0x12 */ errorlib_sub_null, /* 0x13 */ errorlib_sub_null,
	/* 0x14 */ errorlib_sub_null, /* 0x15 */ errorlib_sub_null,
	/* 0x16 */ errorlib_sub_null, /* 0x17 */ errorlib_sub_null,
	/* 0x18 */ errorlib_sub_null, /* 0x19 */ errorlib_sub_null,
	/* 0x1a */ errorlib_sub_null, /* 0x1b */ errorlib_sub_null,
	/* 0x1c */ errorlib_sub_null, /* 0x1d */ errorlib_sub_null,
	/* 0x1e */ errorlib_sub_null, /* 0x1f */ errorlib_sub_null,

	/* 0x20 */ errorlib_sub_null, /* 0x21 */ errorlib_sub_null,
	/* 0x22 */ errorlib_sub_null, /* 0x23 */ errorlib_sub_null,
	/* 0x24 */ errorlib_sub_null, /* 0x25 */ errorlib_sub_null,
	/* 0x26 */ errorlib_sub_null, /* 0x27 */ errorlib_sub_null,
	/* 0x28 */ errorlib_sub_null, /* 0x29 */ errorlib_sub_null,
	/* 0x2a */ errorlib_sub_null, /* 0x2b */ errorlib_sub_null,
	/* 0x2c */ errorlib_sub_null, /* 0x2d */ errorlib_sub_null,
	/* 0x2e */ errorlib_sub_null, /* 0x2f */ errorlib_sub_null,

	/* 0x30 */ errorlib_sub_null, /* 0x31 */ errorlib_sub_null,
	/* 0x32 */ errorlib_sub_null, /* 0x33 */ errorlib_sub_null,
	/* 0x34 */ errorlib_sub_null, /* 0x35 */ errorlib_sub_null,
	/* 0x36 */ errorlib_sub_null,

#if !TARGET_OS_DRIVERKIT
	/* 0x37; err_libkern */
	{
		.max_sub = errlib_count(err_libkern_sub),
		.bad_sub = "(libkern/?) unknown subsystem error",
		.subsystem = err_libkern_sub,
	},

	// /* 0x38; err_iokit */
	// {
	// 	.max_sub = errlib_count(err_iokit_sub),
	// 	.bad_sub = "(iokit/?) unknown subsystem error",
	// 	.subsystem = err_iokit_sub,
	// 	.map_table = err_iokit_sub_map,
	// 	.map_count = errlib_count(err_iokit_sub_map)
	// },
#else
	/* 0x37 */ errorlib_sub_null, /* 0x38 */ errorlib_sub_null,
#endif // TARGET_OS_DRIVERKIT

	/* 0x39 */ errorlib_sub_null,
	/* 0x3a */ errorlib_sub_null, /* 0x3b */ errorlib_sub_null,
	/* 0x3c */ errorlib_sub_null, /* 0x3d */ errorlib_sub_null,
	/* 0x3e */ errorlib_sub_null, /* 0x3f */ errorlib_sub_null,
};

// int error_system_count = errlib_count(_mach_errors);
