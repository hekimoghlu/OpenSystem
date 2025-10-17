/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 17, 2022.
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
#ifndef __INCLUDE_UAPI_SOUND_SOF_USER_HEADER_H__
#define __INCLUDE_UAPI_SOUND_SOF_USER_HEADER_H__
#include <linux/types.h>
struct sof_abi_hdr {
  __u32 magic;
  __u32 type;
  __u32 size;
  __u32 abi;
  __u32 reserved[4];
  __u32 data[];
} __attribute__((__packed__));
#define SOF_MANIFEST_DATA_TYPE_NHLT 1
struct sof_manifest_tlv {
  __le32 type;
  __le32 size;
  __u8 data[];
};
struct sof_manifest {
  __le16 abi_major;
  __le16 abi_minor;
  __le16 abi_patch;
  __le16 count;
  struct sof_manifest_tlv items[];
};
#endif
