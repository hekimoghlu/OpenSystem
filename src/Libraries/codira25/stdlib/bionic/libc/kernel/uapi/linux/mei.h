/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 25, 2022.
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
#ifndef _LINUX_MEI_H
#define _LINUX_MEI_H
#include <linux/mei_uuid.h>
#define IOCTL_MEI_CONNECT_CLIENT _IOWR('H', 0x01, struct mei_connect_client_data)
struct mei_client {
  __u32 max_msg_length;
  __u8 protocol_version;
  __u8 reserved[3];
};
struct mei_connect_client_data {
  union {
    uuid_le in_client_uuid;
    struct mei_client out_client_properties;
  };
};
#define IOCTL_MEI_NOTIFY_SET _IOW('H', 0x02, __u32)
#define IOCTL_MEI_NOTIFY_GET _IOR('H', 0x03, __u32)
struct mei_connect_client_vtag {
  uuid_le in_client_uuid;
  __u8 vtag;
  __u8 reserved[3];
};
struct mei_connect_client_data_vtag {
  union {
    struct mei_connect_client_vtag connect;
    struct mei_client out_client_properties;
  };
};
#define IOCTL_MEI_CONNECT_CLIENT_VTAG _IOWR('H', 0x04, struct mei_connect_client_data_vtag)
#endif
