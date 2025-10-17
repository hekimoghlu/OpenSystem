/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 20, 2021.
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
#ifndef MACHINE_ARM64_REMOTE_TIME_H
#define MACHINE_ARM64_REMOTE_TIME_H

#include <stdint.h>
#include <sys/cdefs.h>

__BEGIN_DECLS
void mach_bridge_recv_timestamps(uint64_t bridgeTimestamp, uint64_t localTimestamp);
void mach_bridge_init_timestamp(void);
void mach_bridge_set_params(uint64_t local_timestamp, uint64_t remote_timestamp, double rate);
__END_DECLS

#endif /* MACHINE_ARM64_REMOTE_TIME_H */
