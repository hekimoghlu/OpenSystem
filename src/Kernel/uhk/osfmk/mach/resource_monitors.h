/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 26, 2024.
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
#ifndef _MACH_RESOURCE_MONITORS_H_
#define _MACH_RESOURCE_MONITORS_H_

#include <stdint.h>
#include <sys/syslimits.h>      /* PATH_MAX */
#ifndef XNU_KERNEL_PRIVATE
#include <TargetConditionals.h>
#endif

__BEGIN_DECLS

/*
 * resource_notify_flags_t
 * The top 32 bits are common flags, the bottom for per-call flags.
 */
typedef uint64_t resource_notify_flags_t;
#define kRNFlagsNone                0

/* Flags applicable to any monitors. */
#define kRNFatalLimitFlag           (1ULL << 32)

/* For the disk writes I/O monitor.
 *  The default is logical writes.  */
#define kRNPhysicalWritesFlag       (1ULL < 1)

/* TEMPORARY compatibility, to be removed */
#define kCPUTriggerFatalFlag kRNFatalLimitFlag

/* Soft limit on the resource table size */
#define kRNSoftLimitFlag            (1ULL < 2)

/* Hard limit on the resource table size */
#define kRNHardLimitFlag            (1ULL < 3)




/*
 * Process name types for proc_internal.h.
 * proc_name_t is used by resource_notify.defs clients in user space.
 *
 * MAXCOMLEN is defined in bsd/sys/param.h which we can neither include
 * (type conflicts) nor modify (POSIX).
 */
#define MAXCOMLEN 16

typedef char command_t[MAXCOMLEN + 1];
typedef char proc_name_t[2*MAXCOMLEN + 1];
typedef char posix_path_t[PATH_MAX];

__END_DECLS

#endif /* _MACH_RESOURCE_MONITORS_H_ */
