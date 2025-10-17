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
#pragma once

#include <mach/message.h>

#if PLATFORM(MAC) || USE(APPLE_INTERNAL_SDK)

#include <servers/bootstrap.h>

#else

typedef char name_t[128];

#endif

#if USE(APPLE_INTERNAL_SDK)

#include <bootstrap_priv.h>

#endif

WTF_EXTERN_C_BEGIN

kern_return_t bootstrap_look_up(mach_port_t, const name_t serviceName, mach_port_t *);
kern_return_t bootstrap_register2(mach_port_t, name_t, mach_port_t, uint64_t flags);

WTF_EXTERN_C_END
