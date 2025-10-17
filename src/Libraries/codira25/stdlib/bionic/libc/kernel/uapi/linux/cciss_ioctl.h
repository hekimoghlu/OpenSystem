/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 7, 2025.
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
#ifndef _UAPICCISS_IOCTLH
#define _UAPICCISS_IOCTLH
#include <linux/types.h>
#include <linux/ioctl.h>
#include <linux/cciss_defs.h>
#define CCISS_IOC_MAGIC 'B'
typedef struct _cciss_pci_info_struct {
  unsigned char bus;
  unsigned char dev_fn;
  unsigned short domain;
  __u32 board_id;
} cciss_pci_info_struct;
typedef struct _cciss_coalint_struct {
  __u32 delay;
  __u32 count;
} cciss_coalint_struct;
typedef char NodeName_type[16];
typedef __u32 Heartbeat_type;
#define CISS_PARSCSIU2 0x0001
#define CISS_PARCSCIU3 0x0002
#define CISS_FIBRE1G 0x0100
#define CISS_FIBRE2G 0x0200
typedef __u32 BusTypes_type;
typedef char FirmwareVer_type[4];
typedef __u32 DriverVer_type;
#define MAX_KMALLOC_SIZE 128000
typedef struct _IOCTL_Command_struct {
  LUNAddr_struct LUN_info;
  RequestBlock_struct Request;
  ErrorInfo_struct error_info;
  WORD buf_size;
  BYTE  * buf;
} IOCTL_Command_struct;
typedef struct _BIG_IOCTL_Command_struct {
  LUNAddr_struct LUN_info;
  RequestBlock_struct Request;
  ErrorInfo_struct error_info;
  DWORD malloc_size;
  DWORD buf_size;
  BYTE  * buf;
} BIG_IOCTL_Command_struct;
typedef struct _LogvolInfo_struct {
  __u32 LunID;
  int num_opens;
  int num_parts;
} LogvolInfo_struct;
#define CCISS_GETPCIINFO _IOR(CCISS_IOC_MAGIC, 1, cciss_pci_info_struct)
#define CCISS_GETINTINFO _IOR(CCISS_IOC_MAGIC, 2, cciss_coalint_struct)
#define CCISS_SETINTINFO _IOW(CCISS_IOC_MAGIC, 3, cciss_coalint_struct)
#define CCISS_GETNODENAME _IOR(CCISS_IOC_MAGIC, 4, NodeName_type)
#define CCISS_SETNODENAME _IOW(CCISS_IOC_MAGIC, 5, NodeName_type)
#define CCISS_GETHEARTBEAT _IOR(CCISS_IOC_MAGIC, 6, Heartbeat_type)
#define CCISS_GETBUSTYPES _IOR(CCISS_IOC_MAGIC, 7, BusTypes_type)
#define CCISS_GETFIRMVER _IOR(CCISS_IOC_MAGIC, 8, FirmwareVer_type)
#define CCISS_GETDRIVVER _IOR(CCISS_IOC_MAGIC, 9, DriverVer_type)
#define CCISS_REVALIDVOLS _IO(CCISS_IOC_MAGIC, 10)
#define CCISS_PASSTHRU _IOWR(CCISS_IOC_MAGIC, 11, IOCTL_Command_struct)
#define CCISS_DEREGDISK _IO(CCISS_IOC_MAGIC, 12)
#define CCISS_REGNEWDISK _IOW(CCISS_IOC_MAGIC, 13, int)
#define CCISS_REGNEWD _IO(CCISS_IOC_MAGIC, 14)
#define CCISS_RESCANDISK _IO(CCISS_IOC_MAGIC, 16)
#define CCISS_GETLUNINFO _IOR(CCISS_IOC_MAGIC, 17, LogvolInfo_struct)
#define CCISS_BIG_PASSTHRU _IOWR(CCISS_IOC_MAGIC, 18, BIG_IOCTL_Command_struct)
#endif
