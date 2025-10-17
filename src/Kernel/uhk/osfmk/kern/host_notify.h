/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 17, 2022.
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
 * Copyright (c) 2003 Apple Computer, Inc.  All rights reserved.
 *
 * HISTORY
 *
 * 14 January 2003 (debo)
 *  Created.
 */

#ifndef _KERN_HOST_NOTIFY_H_
#define _KERN_HOST_NOTIFY_H_

#ifdef MACH_KERNEL_PRIVATE
#include <mach/mach_types.h>

typedef struct host_notify_entry       *host_notify_t;

void    host_notify_calendar_change(void);
void    host_notify_calendar_set(void);
void    host_notify_cancel(host_notify_t entry);

#endif /* MACH_KERNEL_PRIVATE */

#endif /* _KERN_HOST_NOTIFY_H_ */
