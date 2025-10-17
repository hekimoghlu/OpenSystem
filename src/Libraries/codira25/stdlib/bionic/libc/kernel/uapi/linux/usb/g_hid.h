/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 7, 2022.
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
#ifndef __UAPI_LINUX_USB_G_HID_H
#define __UAPI_LINUX_USB_G_HID_H
#include <linux/types.h>
#define MAX_REPORT_LENGTH 64
struct usb_hidg_report {
  __u8 report_id;
  __u8 userspace_req;
  __u16 length;
  __u8 data[MAX_REPORT_LENGTH];
  __u8 padding[4];
};
#define GADGET_HID_READ_GET_REPORT_ID _IOR('g', 0x41, __u8)
#define GADGET_HID_WRITE_GET_REPORT _IOW('g', 0x42, struct usb_hidg_report)
#endif
