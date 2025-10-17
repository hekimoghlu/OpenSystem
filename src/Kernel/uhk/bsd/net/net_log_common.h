/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 3, 2022.
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
#ifndef _NET_LOG_COMMON_H_
#define _NET_LOG_COMMON_H_

#include <os/log.h>

#define NET_LOG_SUBSYSTEM_PREFIX "com.apple.xnu.net"

#define NET_LOG_SUBSYSTEM_MPTCP         NET_LOG_SUBSYSTEM_PREFIX ".mptcp"
#define NET_LOG_SUBSYSTEM_TCP           NET_LOG_SUBSYSTEM_PREFIX ".tcp"

#endif /* _NET_LOG_COMMON_H_ */
