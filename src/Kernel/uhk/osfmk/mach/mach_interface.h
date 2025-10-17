/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 15, 2025.
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
#ifndef _MACH_INTERFACE_H_
#define _MACH_INTERFACE_H_

#include <mach/clock.h>
#include <mach/clock_priv.h>
#include <mach/clock_reply_server.h>
#include <mach/exc_server.h>
#include <mach/host_priv.h>
#include <mach/host_security.h>
#include <mach/mach_exc_server.h>
#include <mach/mach_host.h>
#include <mach/mach_port.h>
#include <mach/notify_server.h>
#include <mach/processor.h>
#include <mach/processor_set.h>
#include <mach/semaphore.h>
#include <mach/task.h>
#include <mach/thread_act.h>
#include <mach/vm_map.h>

#ifdef XNU_KERNEL_PRIVATE
/*
 * Raw EMMI interfaces are private to xnu
 * and subject to change.
 */
#include <mach/upl.h>
#endif

#endif /* _MACH_INTERFACE_H_ */
