/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 24, 2024.
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
#ifndef _EAPOLCONTROLLER_EXT_H
#define _EAPOLCONTROLLER_EXT_H

#include <mach/mach_init.h>
#include <servers/bootstrap.h>
#include <bootstrap_priv.h>

#define EAPOLCONTROLLER_SERVER		"com.apple.network.EAPOLController"

static __inline__ kern_return_t
eapolcontroller_server_port(mach_port_t * server)
{
    return (bootstrap_look_up2(bootstrap_port, EAPOLCONTROLLER_SERVER, server,
                               0, BOOTSTRAP_PRIVILEGED_SERVER));
}

#endif /* _EAPOLCONTROLLER_EXT_H */
