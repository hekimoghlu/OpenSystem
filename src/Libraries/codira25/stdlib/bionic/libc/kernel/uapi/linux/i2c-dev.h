/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 9, 2021.
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
#ifndef _UAPI_LINUX_I2C_DEV_H
#define _UAPI_LINUX_I2C_DEV_H
#include <linux/types.h>
#include <linux/compiler.h>
#define I2C_RETRIES 0x0701
#define I2C_TIMEOUT 0x0702
#define I2C_SLAVE 0x0703
#define I2C_SLAVE_FORCE 0x0706
#define I2C_TENBIT 0x0704
#define I2C_FUNCS 0x0705
#define I2C_RDWR 0x0707
#define I2C_PEC 0x0708
#define I2C_SMBUS 0x0720
struct i2c_smbus_ioctl_data {
  __u8 read_write;
  __u8 command;
  __u32 size;
  union i2c_smbus_data  * data;
};
struct i2c_rdwr_ioctl_data {
  struct i2c_msg  * msgs;
  __u32 nmsgs;
};
#define I2C_RDWR_IOCTL_MAX_MSGS 42
#define I2C_RDRW_IOCTL_MAX_MSGS I2C_RDWR_IOCTL_MAX_MSGS
#endif
