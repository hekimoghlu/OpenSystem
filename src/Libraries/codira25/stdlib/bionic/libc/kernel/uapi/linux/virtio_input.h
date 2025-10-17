/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 11, 2021.
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
#ifndef _LINUX_VIRTIO_INPUT_H
#define _LINUX_VIRTIO_INPUT_H
#include <linux/types.h>
enum virtio_input_config_select {
  VIRTIO_INPUT_CFG_UNSET = 0x00,
  VIRTIO_INPUT_CFG_ID_NAME = 0x01,
  VIRTIO_INPUT_CFG_ID_SERIAL = 0x02,
  VIRTIO_INPUT_CFG_ID_DEVIDS = 0x03,
  VIRTIO_INPUT_CFG_PROP_BITS = 0x10,
  VIRTIO_INPUT_CFG_EV_BITS = 0x11,
  VIRTIO_INPUT_CFG_ABS_INFO = 0x12,
};
struct virtio_input_absinfo {
  __le32 min;
  __le32 max;
  __le32 fuzz;
  __le32 flat;
  __le32 res;
};
struct virtio_input_devids {
  __le16 bustype;
  __le16 vendor;
  __le16 product;
  __le16 version;
};
struct virtio_input_config {
  __u8 select;
  __u8 subsel;
  __u8 size;
  __u8 reserved[5];
  union {
    char string[128];
    __u8 bitmap[128];
    struct virtio_input_absinfo abs;
    struct virtio_input_devids ids;
  } u;
};
struct virtio_input_event {
  __le16 type;
  __le16 code;
  __le32 value;
};
#endif
