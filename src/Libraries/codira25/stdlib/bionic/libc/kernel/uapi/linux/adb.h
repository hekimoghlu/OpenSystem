/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 27, 2022.
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
#ifndef _UAPI__ADB_H
#define _UAPI__ADB_H
#define ADB_BUSRESET 0
#define ADB_FLUSH(id) (0x01 | ((id) << 4))
#define ADB_WRITEREG(id,reg) (0x08 | (reg) | ((id) << 4))
#define ADB_READREG(id,reg) (0x0C | (reg) | ((id) << 4))
#define ADB_DONGLE 1
#define ADB_KEYBOARD 2
#define ADB_MOUSE 3
#define ADB_TABLET 4
#define ADB_MODEM 5
#define ADB_MISC 7
#define ADB_RET_OK 0
#define ADB_RET_TIMEOUT 3
#define ADB_PACKET 0
#define CUDA_PACKET 1
#define ERROR_PACKET 2
#define TIMER_PACKET 3
#define POWER_PACKET 4
#define MACIIC_PACKET 5
#define PMU_PACKET 6
#define ADB_QUERY 7
#define ADB_QUERY_GETDEVINFO 1
#endif
