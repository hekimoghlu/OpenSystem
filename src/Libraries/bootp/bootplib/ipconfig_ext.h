/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 16, 2023.
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
#ifndef _S_IPCONFIG_EXT_H
#define _S_IPCONFIG_EXT_H

#define IPCONFIG_SERVER		"com.apple.network.IPConfiguration"

#include <mach/mach_init.h>
#include <servers/bootstrap.h>

static __inline__ kern_return_t
ipconfig_server_port(mach_port_t * server)
{
    return (bootstrap_look_up(bootstrap_port, IPCONFIG_SERVER, server));
}
#endif /* _S_IPCONFIG_EXT_H */
