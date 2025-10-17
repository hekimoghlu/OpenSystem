/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 14, 2022.
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
#ifndef __LINUX_CAPI_H__
#define __LINUX_CAPI_H__
#include <linux/types.h>
#include <linux/ioctl.h>
#include <linux/kernelcapi.h>
typedef struct capi_register_params {
  __u32 level3cnt;
  __u32 datablkcnt;
  __u32 datablklen;
} capi_register_params;
#define CAPI_REGISTER _IOW('C', 0x01, struct capi_register_params)
#define CAPI_MANUFACTURER_LEN 64
#define CAPI_GET_MANUFACTURER _IOWR('C', 0x06, int)
typedef struct capi_version {
  __u32 majorversion;
  __u32 minorversion;
  __u32 majormanuversion;
  __u32 minormanuversion;
} capi_version;
#define CAPI_GET_VERSION _IOWR('C', 0x07, struct capi_version)
#define CAPI_SERIAL_LEN 8
#define CAPI_GET_SERIAL _IOWR('C', 0x08, int)
typedef struct capi_profile {
  __u16 ncontroller;
  __u16 nbchannel;
  __u32 goptions;
  __u32 support1;
  __u32 support2;
  __u32 support3;
  __u32 reserved[6];
  __u32 manu[5];
} capi_profile;
#define CAPI_GET_PROFILE _IOWR('C', 0x09, struct capi_profile)
typedef struct capi_manufacturer_cmd {
  unsigned long cmd;
  void  * data;
} capi_manufacturer_cmd;
#define CAPI_MANUFACTURER_CMD _IOWR('C', 0x20, struct capi_manufacturer_cmd)
#define CAPI_GET_ERRCODE _IOR('C', 0x21, __u16)
#define CAPI_INSTALLED _IOR('C', 0x22, __u16)
typedef union capi_ioctl_struct {
  __u32 contr;
  capi_register_params rparams;
  __u8 manufacturer[CAPI_MANUFACTURER_LEN];
  capi_version version;
  __u8 serial[CAPI_SERIAL_LEN];
  capi_profile profile;
  capi_manufacturer_cmd cmd;
  __u16 errcode;
} capi_ioctl_struct;
#define CAPIFLAG_HIGHJACKING 0x0001
#define CAPI_GET_FLAGS _IOR('C', 0x23, unsigned)
#define CAPI_SET_FLAGS _IOR('C', 0x24, unsigned)
#define CAPI_CLR_FLAGS _IOR('C', 0x25, unsigned)
#define CAPI_NCCI_OPENCOUNT _IOR('C', 0x26, unsigned)
#define CAPI_NCCI_GETUNIT _IOR('C', 0x27, unsigned)
#endif
