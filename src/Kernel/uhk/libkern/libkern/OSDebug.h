/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 10, 2024.
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
 * Copyright (c) 1999 Apple Computer, Inc.  All rights reserved.
 *
 * HISTORY
 *
 */

#ifndef _OS_OSDEBBUG_H
#define _OS_OSDEBBUG_H

#include <sys/cdefs.h>
#include <mach/mach_types.h>

__BEGIN_DECLS

/* Report a message with a 4 entry backtrace - very slow */
extern void OSReportWithBacktrace(const char *str, ...) __printflike(1, 2);
extern unsigned OSBacktrace(void **bt, unsigned maxAddrs);

/* Simple dump of 20 backtrace entries */
extern void OSPrintBacktrace(void);

/*! @function OSKernelStackRemaining
 *   @abstract Returns bytes available below the current stack frame.
 *   @discussion Returns bytes available below the current stack frame. Safe for interrupt or thread context.
 *   @result Approximate byte count available. */

vm_offset_t OSKernelStackRemaining( void );

__END_DECLS

#endif /* !_OS_OSDEBBUG_H */
