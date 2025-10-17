/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 2, 2022.
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
/* Copyright (c) 1995 NeXT Computer, Inc. All Rights Reserved */
/*
 * Copyright (c) 1987,1988,1989 Carnegie-Mellon University All rights reserved.
 */
#ifndef _SYS_NETPORT_H_
#define _SYS_NETPORT_H_

#include <_types/_uint32_t.h> /* uint32_t */

typedef uint32_t        netaddr_t;

/*
 * Network Port structure.
 */
typedef struct {
	long        np_uid_high;
	long        np_uid_low;
} np_uid_t;

typedef struct {
	netaddr_t   np_receiver;
	netaddr_t   np_owner;
	np_uid_t    np_puid;
	np_uid_t    np_sid;
} network_port_t;

#endif  /* !_SYS_NETPORT_H_ */
