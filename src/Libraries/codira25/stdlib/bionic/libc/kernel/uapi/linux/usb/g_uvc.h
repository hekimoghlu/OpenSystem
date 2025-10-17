/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 2, 2025.
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
#ifndef __LINUX_USB_G_UVC_H
#define __LINUX_USB_G_UVC_H
#include <linux/ioctl.h>
#include <linux/types.h>
#include <linux/usb/ch9.h>
#define UVC_EVENT_FIRST (V4L2_EVENT_PRIVATE_START + 0)
#define UVC_EVENT_CONNECT (V4L2_EVENT_PRIVATE_START + 0)
#define UVC_EVENT_DISCONNECT (V4L2_EVENT_PRIVATE_START + 1)
#define UVC_EVENT_STREAMON (V4L2_EVENT_PRIVATE_START + 2)
#define UVC_EVENT_STREAMOFF (V4L2_EVENT_PRIVATE_START + 3)
#define UVC_EVENT_SETUP (V4L2_EVENT_PRIVATE_START + 4)
#define UVC_EVENT_DATA (V4L2_EVENT_PRIVATE_START + 5)
#define UVC_EVENT_LAST (V4L2_EVENT_PRIVATE_START + 5)
#define UVC_STRING_CONTROL_IDX 0
#define UVC_STRING_STREAMING_IDX 1
struct uvc_request_data {
  __s32 length;
  __u8 data[60];
};
struct uvc_event {
  union {
    enum usb_device_speed speed;
    struct usb_ctrlrequest req;
    struct uvc_request_data data;
  };
};
#define UVCIOC_SEND_RESPONSE _IOW('U', 1, struct uvc_request_data)
#endif
