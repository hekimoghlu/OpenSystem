/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 22, 2022.
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
#ifndef _UAPINFS_IDMAP_H
#define _UAPINFS_IDMAP_H
#include <linux/types.h>
#define IDMAP_NAMESZ 128
#define IDMAP_TYPE_USER 0
#define IDMAP_TYPE_GROUP 1
#define IDMAP_CONV_IDTONAME 0
#define IDMAP_CONV_NAMETOID 1
#define IDMAP_STATUS_INVALIDMSG 0x01
#define IDMAP_STATUS_AGAIN 0x02
#define IDMAP_STATUS_LOOKUPFAIL 0x04
#define IDMAP_STATUS_SUCCESS 0x08
struct idmap_msg {
  __u8 im_type;
  __u8 im_conv;
  char im_name[IDMAP_NAMESZ];
  __u32 im_id;
  __u8 im_status;
};
#endif
