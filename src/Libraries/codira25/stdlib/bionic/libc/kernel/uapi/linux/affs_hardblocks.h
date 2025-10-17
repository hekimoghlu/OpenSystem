/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 8, 2025.
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
#ifndef AFFS_HARDBLOCKS_H
#define AFFS_HARDBLOCKS_H
#include <linux/types.h>
struct RigidDiskBlock {
  __be32 rdb_ID;
  __be32 rdb_SummedLongs;
  __be32 rdb_ChkSum;
  __be32 rdb_HostID;
  __be32 rdb_BlockBytes;
  __be32 rdb_Flags;
  __be32 rdb_BadBlockList;
  __be32 rdb_PartitionList;
  __be32 rdb_FileSysHeaderList;
  __be32 rdb_DriveInit;
  __be32 rdb_Reserved1[6];
  __be32 rdb_Cylinders;
  __be32 rdb_Sectors;
  __be32 rdb_Heads;
  __be32 rdb_Interleave;
  __be32 rdb_Park;
  __be32 rdb_Reserved2[3];
  __be32 rdb_WritePreComp;
  __be32 rdb_ReducedWrite;
  __be32 rdb_StepRate;
  __be32 rdb_Reserved3[5];
  __be32 rdb_RDBBlocksLo;
  __be32 rdb_RDBBlocksHi;
  __be32 rdb_LoCylinder;
  __be32 rdb_HiCylinder;
  __be32 rdb_CylBlocks;
  __be32 rdb_AutoParkSeconds;
  __be32 rdb_HighRDSKBlock;
  __be32 rdb_Reserved4;
  char rdb_DiskVendor[8];
  char rdb_DiskProduct[16];
  char rdb_DiskRevision[4];
  char rdb_ControllerVendor[8];
  char rdb_ControllerProduct[16];
  char rdb_ControllerRevision[4];
  __be32 rdb_Reserved5[10];
};
#define IDNAME_RIGIDDISK 0x5244534B
struct PartitionBlock {
  __be32 pb_ID;
  __be32 pb_SummedLongs;
  __be32 pb_ChkSum;
  __be32 pb_HostID;
  __be32 pb_Next;
  __be32 pb_Flags;
  __be32 pb_Reserved1[2];
  __be32 pb_DevFlags;
  __u8 pb_DriveName[32];
  __be32 pb_Reserved2[15];
  __be32 pb_Environment[17];
  __be32 pb_EReserved[15];
};
#define IDNAME_PARTITION 0x50415254
#define RDB_ALLOCATION_LIMIT 16
#endif
