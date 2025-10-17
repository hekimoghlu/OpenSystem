/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 4, 2025.
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
#ifndef __GETHOSTUUID_PRIVATE_H
#define __GETHOSTUUID_PRIVATE_H

#include <gethostuuid.h>
#include <sys/_types/_timespec.h>
#include <sys/_types/_uuid_t.h>
#include <sys/cdefs.h>
#include <Availability.h>

__BEGIN_DECLS
/* SPI prototype, TEMPORARY */
int _getprivatesystemidentifier(uuid_t uuid, const struct timespec *timeout) __OSX_AVAILABLE_STARTING(__MAC_10_9, __IPHONE_7_0);

/* Callback should return -1 and set errno on failure */
int _register_gethostuuid_callback(int (*)(uuid_t)) __OSX_AVAILABLE_STARTING(__MAC_10_9, __IPHONE_7_0);
__END_DECLS

#endif /* __GETHOSTUUID_PRIVATE_H */
