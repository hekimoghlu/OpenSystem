/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 20, 2022.
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
#ifndef _NETINET_IN_RESTRICTED_PORT_H_
#define _NETINET_IN_RESTRICTED_PORT_H_

#ifdef BSD_KERNEL_PRIVATE

#include <kern/bits.h>

#define PORT_FLAGS_LISTENER 0x00
#if SKYWALK
#define PORT_FLAGS_SKYWALK      0x01
#endif /* SKYWALK */
#define PORT_FLAGS_BSD      0x02
#define PORT_FLAGS_PF       0x03
#define PORT_FLAGS_MAX      0x03

/*
 * the port in network byte order
 */
#define IS_RESTRICTED_IN_PORT(x) (bitmap_test(restricted_port_bitmap, ntohs((uint16_t)(x))))

extern bitmap_t *__sized_by_or_null(BITMAP_SIZE(UINT16_MAX)) restricted_port_bitmap;

extern void restricted_in_port_init(void);

/*
 * The port must be in network byte order
 */
extern bool current_task_can_use_restricted_in_port(in_port_t port, uint8_t protocol, uint32_t port_flags);

#endif /* BSD_KERNEL_PRIVATE */

#endif /* _NETINET_IN_RESTRICTED_PORT_H_ */
