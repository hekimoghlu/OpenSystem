/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 3, 2024.
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
#ifndef _UAPI_LINUX_SURFACE_AGGREGATOR_CDEV_H
#define _UAPI_LINUX_SURFACE_AGGREGATOR_CDEV_H
#include <linux/ioctl.h>
#include <linux/types.h>
enum ssam_cdev_request_flags {
  SSAM_CDEV_REQUEST_HAS_RESPONSE = 0x01,
  SSAM_CDEV_REQUEST_UNSEQUENCED = 0x02,
};
struct ssam_cdev_request {
  __u8 target_category;
  __u8 target_id;
  __u8 command_id;
  __u8 instance_id;
  __u16 flags;
  __s16 status;
  struct {
    __u64 data;
    __u16 length;
    __u8 __pad[6];
  } payload;
  struct {
    __u64 data;
    __u16 length;
    __u8 __pad[6];
  } response;
} __attribute__((__packed__));
struct ssam_cdev_notifier_desc {
  __s32 priority;
  __u8 target_category;
} __attribute__((__packed__));
struct ssam_cdev_event_desc {
  struct {
    __u8 target_category;
    __u8 target_id;
    __u8 cid_enable;
    __u8 cid_disable;
  } reg;
  struct {
    __u8 target_category;
    __u8 instance;
  } id;
  __u8 flags;
} __attribute__((__packed__));
struct ssam_cdev_event {
  __u8 target_category;
  __u8 target_id;
  __u8 command_id;
  __u8 instance_id;
  __u16 length;
  __u8 data[];
} __attribute__((__packed__));
#define SSAM_CDEV_REQUEST _IOWR(0xA5, 1, struct ssam_cdev_request)
#define SSAM_CDEV_NOTIF_REGISTER _IOW(0xA5, 2, struct ssam_cdev_notifier_desc)
#define SSAM_CDEV_NOTIF_UNREGISTER _IOW(0xA5, 3, struct ssam_cdev_notifier_desc)
#define SSAM_CDEV_EVENT_ENABLE _IOW(0xA5, 4, struct ssam_cdev_event_desc)
#define SSAM_CDEV_EVENT_DISABLE _IOW(0xA5, 5, struct ssam_cdev_event_desc)
#endif
