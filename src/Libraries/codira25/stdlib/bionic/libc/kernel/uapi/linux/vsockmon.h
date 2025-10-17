/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 16, 2022.
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
#ifndef _UAPI_VSOCKMON_H
#define _UAPI_VSOCKMON_H
#include <linux/virtio_vsock.h>
struct af_vsockmon_hdr {
  __le64 src_cid;
  __le64 dst_cid;
  __le32 src_port;
  __le32 dst_port;
  __le16 op;
  __le16 transport;
  __le16 len;
  __u8 reserved[2];
};
enum af_vsockmon_op {
  AF_VSOCK_OP_UNKNOWN = 0,
  AF_VSOCK_OP_CONNECT = 1,
  AF_VSOCK_OP_DISCONNECT = 2,
  AF_VSOCK_OP_CONTROL = 3,
  AF_VSOCK_OP_PAYLOAD = 4,
};
enum af_vsockmon_transport {
  AF_VSOCK_TRANSPORT_UNKNOWN = 0,
  AF_VSOCK_TRANSPORT_NO_INFO = 1,
  AF_VSOCK_TRANSPORT_VIRTIO = 2,
};
#endif
