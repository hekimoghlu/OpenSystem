/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 31, 2025.
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
#ifndef _PEXPERT_ARM64_APT_MSG_H
#define _PEXPERT_ARM64_APT_MSG_H

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#include <stdint.h>

#define APT_MSG_NS_KERN 0
#define APT_MSG_KERN_CSWITCH_TIME 0

void apt_msg_init(void);
uint8_t apt_msg_policy(void);
void apt_msg_init_cpu(void);
void apt_msg_emit(int ns, int type, int num_payloads, uint64_t *payloads);

#ifdef __cplusplus
}
#endif /* __cplusplus */
#endif /* _PEXPERT_ARM64_APT_MSG_H */
