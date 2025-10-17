/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 12, 2023.
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
#ifndef __LINUX_USB_GADGETFS_H
#define __LINUX_USB_GADGETFS_H
#include <linux/types.h>
#include <linux/ioctl.h>
#include <linux/usb/ch9.h>
enum usb_gadgetfs_event_type {
  GADGETFS_NOP = 0,
  GADGETFS_CONNECT,
  GADGETFS_DISCONNECT,
  GADGETFS_SETUP,
  GADGETFS_SUSPEND,
};
struct usb_gadgetfs_event {
  union {
    enum usb_device_speed speed;
    struct usb_ctrlrequest setup;
  } u;
  enum usb_gadgetfs_event_type type;
};
#define GADGETFS_FIFO_STATUS _IO('g', 1)
#define GADGETFS_FIFO_FLUSH _IO('g', 2)
#define GADGETFS_CLEAR_HALT _IO('g', 3)
#endif
