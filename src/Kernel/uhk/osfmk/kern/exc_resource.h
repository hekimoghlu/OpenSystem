/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 23, 2023.
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
 *        EXC_RESOURCE related macros, namespace etc.
 */

#ifndef _EXC_RESOURCE_H_
#define _EXC_RESOURCE_H_

/*
 * Generic exception code format:
 *
 * code:
 * +----------------------------------------------------------+
 * |[63:61] type | [60:58] flavor | [57:0] type-specific data |
 * +----------------------------------------------------------+
 */


/* EXC_RESOURCE type and flavor decoding routines */
#define EXC_RESOURCE_DECODE_RESOURCE_TYPE(code) \
	(((code) >> 61) & 0x7ULL)
#define EXC_RESOURCE_DECODE_FLAVOR(code) \
	(((code) >> 58) & 0x7ULL)

/* EXC_RESOURCE Types */
#define RESOURCE_TYPE_CPU       1
#define RESOURCE_TYPE_WAKEUPS   2
#define RESOURCE_TYPE_MEMORY    3
#define RESOURCE_TYPE_IO        4
#define RESOURCE_TYPE_THREADS   5
#define RESOURCE_TYPE_PORTS     6

/* RESOURCE_TYPE_CPU flavors */
#define FLAVOR_CPU_MONITOR              1
#define FLAVOR_CPU_MONITOR_FATAL        2

/*
 * RESOURCE_TYPE_CPU exception code & subcode.
 *
 * This is sent by the kernel when the CPU usage monitor
 * is tripped. [See proc_set_cpumon_params()]
 *
 * code:
 * +-----------------------------------------------+
 * |[63:61] RESOURCE |[60:58] FLAVOR_CPU_ |[57:32] |
 * |_TYPE_CPU        |MONITOR[_FATAL]     |Unused  |
 * +-----------------------------------------------+
 * |[31:7]  Interval (sec)    | [6:0] CPU limit (%)|
 * +-----------------------------------------------+
 *
 * subcode:
 * +-----------------------------------------------+
 * |                          | [6:0] % of CPU     |
 * |                          | actually consumed  |
 * +-----------------------------------------------+
 *
 */

/* RESOURCE_TYPE_CPU decoding macros */
#define EXC_RESOURCE_CPUMONITOR_DECODE_INTERVAL(code) \
	(((code) >> 7) & 0x1FFFFFFULL)
#define EXC_RESOURCE_CPUMONITOR_DECODE_PERCENTAGE(code) \
	((code) & 0x7FULL)
#define EXC_RESOURCE_CPUMONITOR_DECODE_PERCENTAGE_OBSERVED(subcode) \
	((subcode) & 0x7FULL)


/* RESOURCE_TYPE_WAKEUPS flavors */
#define FLAVOR_WAKEUPS_MONITOR  1

/*
 * RESOURCE_TYPE_WAKEUPS exception code & subcode.
 *
 * This is sent by the kernel when the platform idle
 * wakeups monitor is tripped.
 * [See proc_set_wakeupsmon_params()]
 *
 * code:
 * +-----------------------------------------------+
 * |[63:61] RESOURCE |[60:58] FLAVOR_     |[57:32] |
 * |_TYPE_WAKEUPS    |WAKEUPS_MONITOR     |Unused  |
 * +-----------------------------------------------+
 * | [31:20] Observation     | [19:0] # of wakeups |
 * |         interval (sec)  | permitted (per sec) |
 * +-----------------------------------------------+
 *
 * subcode:
 * +-----------------------------------------------+
 * |                         | [19:0] # of wakeups |
 * |                         | observed (per sec)  |
 * +-----------------------------------------------+
 *
 */

#define EXC_RESOURCE_CPUMONITOR_DECODE_WAKEUPS_PERMITTED(code) \
	((code) & 0xFFFULL)
#define EXC_RESOURCE_CPUMONITOR_DECODE_OBSERVATION_INTERVAL(code) \
	(((code) >> 20) & 0xFFFFFULL)
#define EXC_RESOURCE_CPUMONITOR_DECODE_WAKEUPS_OBSERVED(subcode) \
	((subcode) & 0xFFFFFULL)

/* RESOURCE_TYPE_MEMORY flavors */
#define FLAVOR_HIGH_WATERMARK   1       /* Indicates that the exception is due to memory limit warning */
#define FLAVOR_DIAG_MEMLIMIT    2       /* Indicates that the exception is due to a preset diagnostics memory consumption threshold  */

/*
 * RESOURCE_TYPE_MEMORY / FLAVOR_HIGH_WATERMARK
 * exception code & subcode.
 *
 * This is sent by the kernel when a task crosses its high
 * watermark memory limit or when a preset memory consumption
 * threshold is crossed.
 *
 * code:
 * +------------------------------------------------+
 * |[63:61] RESOURCE |[60:58] FLAVOR_HIGH_ |[57:32] |
 * |_TYPE_MEMORY     |WATERMARK            |Unused  |
 * +------------------------------------------------+
 * |                         | [12:0] HWM limit (MB)|
 * +------------------------------------------------+
 *
 * subcode:
 * +------------------------------------------------+
 * |                                         unused |
 * +------------------------------------------------+
 *
 */

#define EXC_RESOURCE_HWM_DECODE_LIMIT(code) \
	((code) & 0x1FFFULL)

/* RESOURCE_TYPE_IO flavors */
#define FLAVOR_IO_PHYSICAL_WRITES               1
#define FLAVOR_IO_LOGICAL_WRITES                2

/*
 * RESOURCE_TYPE_IO exception code & subcode.
 *
 * This is sent by the kernel when a task crosses its
 * I/O limits.
 *
 * code:
 * +-----------------------------------------------+
 * |[63:61] RESOURCE |[60:58] FLAVOR_IO_  |[57:32] |
 * |_TYPE_IO         |PHYSICAL/LOGICAL    |Unused  |
 * +-----------------------------------------------+
 * |[31:15]  Interval (sec)    | [14:0] Limit (MB) |
 * +-----------------------------------------------+
 *
 * subcode:
 * +-----------------------------------------------+
 * |                           | [14:0] I/O Count  |
 * |                           | (in MB)           |
 * +-----------------------------------------------+
 *
 */

/* RESOURCE_TYPE_IO decoding macros */
#define EXC_RESOURCE_IO_DECODE_INTERVAL(code) \
	(((code) >> 15) & 0x1FFFFULL)
#define EXC_RESOURCE_IO_DECODE_LIMIT(code) \
	((code) & 0x7FFFULL)
#define EXC_RESOURCE_IO_OBSERVED(subcode) \
	((subcode) & 0x7FFFULL)


/*
 * RESOURCE_TYPE_THREADS exception code & subcode
 *
 * This is sent by the kernel when a task crosses its
 * thread limit.
 */

#define EXC_RESOURCE_THREADS_DECODE_THREADS(code) \
	((code) & 0x7FFFULL)

/* RESOURCE_TYPE_THREADS flavors */
#define FLAVOR_THREADS_HIGH_WATERMARK 1

/* RESOURCE_TYPE_PORTS flavors */
#define FLAVOR_PORT_SPACE_FULL 1

/*
 * RESOURCE_TYPE_PORTS exception code & subcode.
 *
 * This is sent by the kernel when the process is
 * leaking ipc ports and has filled its port space
 *
 * code:
 * +-----------------------------------------------+
 * |[63:61] RESOURCE |[60:58] FLAVOR_     |[57:32] |
 * |_TYPE_PORTS      |PORT_SPACE_FULL      |Unused  |
 * +-----------------------------------------------+
 * | [31:24] Unused          | [23:0] # of ports   |
 * |                         | allocated           |
 * +-----------------------------------------------+
 *
 * subcode:
 * +-----------------------------------------------+
 * |                         | Unused              |
 * |                         |                     |
 * +-----------------------------------------------+
 *
 */
#define EXC_RESOURCE_THREADS_DECODE_PORTS(code) \
	((code) & 0xFFFFFFULL)

#ifdef KERNEL

/* EXC_RESOURCE type and flavor encoding macros */
#define EXC_RESOURCE_ENCODE_TYPE(code, type) \
	((code) |= (((uint64_t)(type) & 0x7ULL) << 61))
#define EXC_RESOURCE_ENCODE_FLAVOR(code, flavor) \
	((code) |= (((uint64_t)(flavor) & 0x7ULL) << 58))

/* RESOURCE_TYPE_CPU::FLAVOR_CPU_MONITOR specific encoding macros */
#define EXC_RESOURCE_CPUMONITOR_ENCODE_INTERVAL(code, interval) \
	((code) |= (((uint64_t)(interval) & 0x1FFFFFFULL) << 7))
#define EXC_RESOURCE_CPUMONITOR_ENCODE_PERCENTAGE(code, percentage) \
	((code) |= (((uint64_t)(percentage) & 0x7FULL)))

/* RESOURCE_TYPE_WAKEUPS::FLAVOR_WAKEUPS_MONITOR specific encoding macros */
#define EXC_RESOURCE_CPUMONITOR_ENCODE_WAKEUPS_PERMITTED(code, num) \
	((code) |= ((uint64_t)(num) & 0xFFFFFULL))
#define EXC_RESOURCE_CPUMONITOR_ENCODE_OBSERVATION_INTERVAL(code, num) \
	((code) |= (((uint64_t)(num) & 0xFFFULL) << 20))
#define EXC_RESOURCE_CPUMONITOR_ENCODE_WAKEUPS_OBSERVED(subcode, num) \
	((subcode) |= ((uint64_t)(num) & 0xFFFFFULL))

/* RESOURCE_TYPE_MEMORY::FLAVOR_HIGH_WATERMARK specific encoding macros */
#define EXC_RESOURCE_HWM_ENCODE_LIMIT(code, num) \
	((code) |= ((uint64_t)(num) & 0x1FFFULL))

/* RESOURCE_TYPE_IO::FLAVOR_IO_PHYSICAL_WRITES/FLAVOR_IO_LOGICAL_WRITES specific encoding macros */
#define EXC_RESOURCE_IO_ENCODE_INTERVAL(code, interval) \
	((code) |= (((uint64_t)(interval) & 0x1FFFFULL) << 15))
#define EXC_RESOURCE_IO_ENCODE_LIMIT(code, limit) \
	((code) |= (((uint64_t)(limit) & 0x7FFFULL)))
#define EXC_RESOURCE_IO_ENCODE_OBSERVED(subcode, num) \
	((subcode) |= (((uint64_t)(num) & 0x7FFFULL)))

/* RESOURCE_TYPE_THREADS specific encoding macros */
#define EXC_RESOURCE_THREADS_ENCODE_THREADS(code, threads) \
	((code) |= (((uint64_t)(threads) & 0x7FFFULL)))

/* RESOURCE_TYPE_PORTS::FLAVOR_PORT_SPACE_FULL specific encoding macros */
#define EXC_RESOURCE_PORTS_ENCODE_PORTS(code, num) \
	((code) |= ((uint64_t)(num) & 0xFFFFFFULL))

#endif /* KERNEL */


#endif /* _EXC_RESOURCE_H_ */
