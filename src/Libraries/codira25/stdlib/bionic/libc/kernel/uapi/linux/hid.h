/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 22, 2024.
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
#ifndef _UAPI__HID_H
#define _UAPI__HID_H
#define USB_INTERFACE_CLASS_HID 3
#define USB_INTERFACE_SUBCLASS_BOOT 1
#define USB_INTERFACE_PROTOCOL_KEYBOARD 1
#define USB_INTERFACE_PROTOCOL_MOUSE 2
enum hid_report_type {
  HID_INPUT_REPORT = 0,
  HID_OUTPUT_REPORT = 1,
  HID_FEATURE_REPORT = 2,
  HID_REPORT_TYPES,
};
enum hid_class_request {
  HID_REQ_GET_REPORT = 0x01,
  HID_REQ_GET_IDLE = 0x02,
  HID_REQ_GET_PROTOCOL = 0x03,
  HID_REQ_SET_REPORT = 0x09,
  HID_REQ_SET_IDLE = 0x0A,
  HID_REQ_SET_PROTOCOL = 0x0B,
};
#define HID_DT_HID (USB_TYPE_CLASS | 0x01)
#define HID_DT_REPORT (USB_TYPE_CLASS | 0x02)
#define HID_DT_PHYSICAL (USB_TYPE_CLASS | 0x03)
#define HID_MAX_DESCRIPTOR_SIZE 4096
#endif
