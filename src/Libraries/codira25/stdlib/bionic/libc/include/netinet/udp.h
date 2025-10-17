/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 27, 2024.
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
#ifndef _NETINET_UDP_H
#define _NETINET_UDP_H

#include <sys/cdefs.h>
#include <sys/types.h>

#include <linux/udp.h>

struct udphdr {
    __extension__ union {
        struct /* BSD names */ {
            u_int16_t uh_sport;
            u_int16_t uh_dport;
            u_int16_t uh_ulen;
            u_int16_t uh_sum;
        };
        struct /* Linux names */ {
            u_int16_t source;
            u_int16_t dest;
            u_int16_t len;
            u_int16_t check;
        };
    };
};

#endif /* _NETINET_UDP_H */
