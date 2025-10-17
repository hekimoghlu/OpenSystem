/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 26, 2023.
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
#ifndef _UAPI_LINUX_CUDA_H
#define _UAPI_LINUX_CUDA_H
#define CUDA_WARM_START 0
#define CUDA_AUTOPOLL 1
#define CUDA_GET_6805_ADDR 2
#define CUDA_GET_TIME 3
#define CUDA_GET_PRAM 7
#define CUDA_SET_6805_ADDR 8
#define CUDA_SET_TIME 9
#define CUDA_POWERDOWN 0xa
#define CUDA_POWERUP_TIME 0xb
#define CUDA_SET_PRAM 0xc
#define CUDA_MS_RESET 0xd
#define CUDA_SEND_DFAC 0xe
#define CUDA_RESET_SYSTEM 0x11
#define CUDA_SET_IPL 0x12
#define CUDA_SET_AUTO_RATE 0x14
#define CUDA_GET_AUTO_RATE 0x16
#define CUDA_SET_DEVICE_LIST 0x19
#define CUDA_GET_DEVICE_LIST 0x1a
#define CUDA_GET_SET_IIC 0x22
#endif
