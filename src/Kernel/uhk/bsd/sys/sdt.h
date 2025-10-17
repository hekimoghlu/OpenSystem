/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 9, 2024.
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
 * Copyright 2004 Sun Microsystems, Inc.  All rights reserved.
 * Use is subject to license terms.
 */

#ifndef _SYS_SDT_H
#define _SYS_SDT_H

/*
 * This is a wrapper header that wraps the mach visible sdt.h header so that
 * the header file ends up visible where software expects it to be.
 *
 * Note:  The process of adding USDT probes to code is slightly different
 * than documented in the "Solaris Dynamic Tracing Guide".
 * The DTRACE_PROBE*() macros are not supported on Mac OS X -- instead see
 * "BUILDING CODE CONTAINING USDT PROBES" in the dtrace(1) manpage
 *
 */
#include <sys/cdefs.h>
#include <mach/sdt.h>

#endif  /* _SYS_SDT_H */
