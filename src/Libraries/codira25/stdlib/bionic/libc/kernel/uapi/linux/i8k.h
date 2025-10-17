/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 5, 2023.
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
#ifndef _LINUX_I8K_H
#define _LINUX_I8K_H
#define I8K_PROC "/proc/i8k"
#define I8K_PROC_FMT "1.0"
#define I8K_BIOS_VERSION _IOR('i', 0x80, int)
#define I8K_MACHINE_ID _IOR('i', 0x81, int)
#define I8K_POWER_STATUS _IOR('i', 0x82, size_t)
#define I8K_FN_STATUS _IOR('i', 0x83, size_t)
#define I8K_GET_TEMP _IOR('i', 0x84, size_t)
#define I8K_GET_SPEED _IOWR('i', 0x85, size_t)
#define I8K_GET_FAN _IOWR('i', 0x86, size_t)
#define I8K_SET_FAN _IOWR('i', 0x87, size_t)
#define I8K_FAN_LEFT 1
#define I8K_FAN_RIGHT 0
#define I8K_FAN_OFF 0
#define I8K_FAN_LOW 1
#define I8K_FAN_HIGH 2
#define I8K_FAN_TURBO 3
#define I8K_FAN_MAX I8K_FAN_TURBO
#define I8K_VOL_UP 1
#define I8K_VOL_DOWN 2
#define I8K_VOL_MUTE 4
#define I8K_AC 1
#define I8K_BATTERY 0
#endif
