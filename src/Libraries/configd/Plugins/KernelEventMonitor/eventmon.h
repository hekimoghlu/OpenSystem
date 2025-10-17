/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 17, 2024.
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
 * Modification History
 *
 * August 5, 2002	Allan Nathanson <ajn@apple.com>
 * - split code out from eventmon.c
 */


#ifndef _EVENTMON_H
#define _EVENTMON_H

#include <sys/cdefs.h>
#include <unistd.h>
#include <sys/types.h>
#define	KERNEL_PRIVATE
#include <sys/sockio.h>
#undef	KERNEL_PRIVATE
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <sys/filio.h>
#include <sys/kern_event.h>
#include <errno.h>
#include <net/if.h>
#include <net/if_dl.h>
#include <net/if_media.h>
#include <net/route.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <ifaddrs.h>
#include <CoreFoundation/CoreFoundation.h>
#include <SystemConfiguration/SystemConfiguration.h>
#include <SystemConfiguration/SCPrivate.h>
#include <SystemConfiguration/SCValidation.h>


extern Boolean			network_changed;
extern SCDynamicStoreRef	store;
extern Boolean			_verbose;


__BEGIN_DECLS

os_log_t
__log_KernelEventMonitor	(void);

int
dgram_socket			(int		domain);

void
messages_add_msg_with_arg	(const char	*msg,
				 const char	*arg);

void
config_new_interface		(const char	*ifname);

__END_DECLS

#endif /* _EVENTMON_H */

