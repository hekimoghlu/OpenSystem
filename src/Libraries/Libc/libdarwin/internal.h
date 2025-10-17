/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 28, 2025.
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
/*!
 * @header
 * libdarwin internal header.
 */
#ifndef __DARWIN_INTERNAL_H
#define __DARWIN_INTERNAL_H

#include <os/base.h>
#include <os/api.h>
#include <Availability.h>

#include <mach/port.h>
#include <mach/message.h>
#include <mach/host_priv.h>
#include <mach/host_reboot.h>
#include <mach/mach_time.h>
#include <mach/mach.h>
#include <mach/port.h>
#include <mach/message.h>
#include <mach/host_priv.h>
#include <mach/host_reboot.h>
#include <mach/kern_return.h>

#include <sys/sysctl.h>
#include <sys/reboot.h>
#include <sys/syscall.h>
#include <sys/errno.h>
#include <sys/paths.h>
#include <sys/spawn.h>
#include <sys/proc_info.h>
#include <sys/sysctl.h>
#include <sys/reboot.h>
#include <sys/syscall.h>
#include <sys/errno.h>
#include <sys/paths.h>
#include <sys/spawn.h>
#include <sys/proc_info.h>
#include <crt_externs.h>

#define OS_CRASH_ENABLE_EXPERIMENTAL_LIBTRACE 1
#include <os/assumes.h>
#include <os/transaction_private.h>
#include <os/log_private.h>
#include <os/alloc_once_private.h>

#include <mach-o/getsect.h>
#include <bsm/libbsm.h>
#include <sysexits.h>
#include <spawn.h>
#include <libproc.h>
#include <string.h>
#include <dlfcn.h>
#include <err.h>
#include <ctype.h>
#include <struct.h>
#include <bootstrap_priv.h>
#include <assert.h>
#include <sys/ioctl.h>

#include "h/bsd.h"
#include "h/cleanup.h"
#include "h/ctl.h"
#include "h/err.h"
#include "h/errno.h"
#include "h/mach_exception.h"
#include "h/mach_utils.h"
#include "h/stdio.h"
#include "h/stdlib.h"
#include "h/string.h"

#endif //__DARWIN_INTERNAL_H
