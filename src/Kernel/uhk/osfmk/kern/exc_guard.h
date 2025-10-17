/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 26, 2024.
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
 * Mach Operating System
 * Copyright (c) 1989 Carnegie-Mellon University
 * Copyright (c) 1988 Carnegie-Mellon University
 * Copyright (c) 1987 Carnegie-Mellon University
 * All rights reserved.  The CMU software License Agreement specifies
 * the terms and conditions for use and redistribution.
 */

/*
 * EXC_GUARD related macros, namespace etc.
 */

#ifndef _EXC_GUARD_H_
#define _EXC_GUARD_H_

/*
 * EXC_GUARD exception code namespace.
 *
 * code:
 * +-------------------+----------------+--------------+
 * |[63:61] guard type | [60:32] flavor | [31:0] target|
 * +-------------------+----------------+--------------+
 *
 * subcode:
 * +---------------------------------------------------+
 * |[63:0] guard identifier                            |
 * +---------------------------------------------------+
 */

#define EXC_GUARD_DECODE_GUARD_TYPE(code) \
	((((uint64_t)(code)) >> 61) & 0x7ull)
#define EXC_GUARD_DECODE_GUARD_FLAVOR(code) \
	((((uint64_t)(code)) >> 32) & 0x1fffffff)
#define EXC_GUARD_DECODE_GUARD_TARGET(code) \
	((uint32_t)(code))

/* EXC_GUARD types */

#define GUARD_TYPE_NONE         0x0

/*
 * Mach port guards use the exception codes like this:
 *
 * code:
 * +-----------------------------+----------------+-----------------+
 * |[63:61] GUARD_TYPE_MACH_PORT | [60:32] flavor | [31:0] target   |
 * +-----------------------------+----------------+-----------------+
 *
 * subcode:
 * +----------------------------------------------------------------+
 * |[63:0] payload                                                  |
 * +----------------------------------------------------------------+
 *
 * - flavors are defined in <mach/port.h>
 * - meaning of target and payload is described
 *   in doc/mach_ipc/guard_exceptions.md
 */

#define GUARD_TYPE_MACH_PORT    0x1      /* guarded mach port */

/*
 * File descriptor guards use the exception codes this:
 *
 * code:
 * +-----------------------------+----------------+-----------------+
 * |[63:61] GUARD_TYPE_FD        | [60:32] flavor | [31:0] fd       |
 * +-----------------------------+----------------+-----------------+
 *
 * subcode:
 * +----------------------------------------------------------------+
 * |[63:0] guard identifier                                         |
 * +----------------------------------------------------------------+
 */

#define GUARD_TYPE_FD           0x2     /* guarded file descriptor */

/*
 * User generated guards use the exception codes this:
 *
 * code:
 * +-----------------------------+----------------+-----------------+
 * |[63:61] GUARD_TYPE_USER      | [60:32] unused | [31:0] namespc  |
 * +-----------------------------+----------------+-----------------+
 *
 * subcode:
 * +----------------------------------------------------------------+
 * |[63:0] reason_code                                              |
 * +----------------------------------------------------------------+
 */

#define GUARD_TYPE_USER         0x3     /* Userland assertions */

/*
 * Vnode guards use the exception codes like this:
 *
 * code:
 * +-----------------------------+----------------+-----------------+
 * |[63:61] GUARD_TYPE_VN        | [60:32] flavor | [31:0] pid      |
 * +-----------------------------+----------------+-----------------+
 *
 * subcode:
 * +----------------------------------------------------------------+
 * |[63:0] guard identifier                                         |
 * +----------------------------------------------------------------+
 */

#define GUARD_TYPE_VN           0x4     /* guarded vnode */

/*
 * VM guards use the exception codes like this:
 *
 * code:
 * +-------------------------------+----------------+-----------------+
 * |[63:61] GUARD_TYPE_VIRT_MEMORY | [60:32] flavor | [31:0] unused   |
 * +-------------------------------+----------------+-----------------+
 *
 * subcode:
 * +----------------------------------------------------------------+
 * |[63:0] offset                                                   |
 * +----------------------------------------------------------------+
 */

#define GUARD_TYPE_VIRT_MEMORY  0x5     /* VM operation violating guard */

/*
 * Rejected syscalls use the exception codes like this:
 *
 * code:
 * +-------------------------------+----------------+------------------+
 * |[63:61] GUARD_TYPE_REJECTED_SC | [60:32] unused | [31:0] mach_trap |
 * +-------------------------------+----------------+------------------+
 *
 * subcode:
 * +----------------------------------------------------------------+
 * |[63:0] syscall (if mach_trap field is 0), or mach trap number   |
 * +----------------------------------------------------------------+
 */

#define GUARD_TYPE_REJECTED_SC  0x6     /* rejected system call trap */

#ifdef KERNEL

#define EXC_GUARD_ENCODE_TYPE(code, type) \
	((code) |= (((uint64_t)(type) & 0x7ull) << 61))
#define EXC_GUARD_ENCODE_FLAVOR(code, flavor) \
	((code) |= (((uint64_t)(flavor) & 0x1fffffffull) << 32))
#define EXC_GUARD_ENCODE_TARGET(code, target) \
	((code) |= (((uint64_t)(target) & 0xffffffffull)))

#endif /* KERNEL */

#endif /* _EXC_GUARD_H_ */
