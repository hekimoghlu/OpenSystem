/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 10, 2022.
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
#ifndef __CCID_USB_H__
#define __CCID_USB_H__
status_t OpenUSB(unsigned int reader_index, int channel);

status_t OpenUSBByName(unsigned int reader_index, /*@null@*/ char *device);

status_t WriteUSB(unsigned int reader_index, unsigned int length,
	unsigned char *Buffer);

status_t ReadUSB(unsigned int reader_index, unsigned int *length,
	/*@out@*/ unsigned char *Buffer, int bSeq);

status_t CloseUSB(unsigned int reader_index);
status_t DisconnectUSB(unsigned int reader_index);

#include <libusb.h>
/*@null@*/ const struct libusb_interface *get_ccid_usb_interface(
	struct libusb_config_descriptor *desc, int *num);

const unsigned char *get_ccid_device_descriptor(const struct libusb_interface *usb_interface);

uint8_t get_ccid_usb_bus_number(int reader_index);
uint8_t get_ccid_usb_device_address(int reader_index);

int ControlUSB(int reader_index, int requesttype, int request, int value,
	unsigned char *bytes, unsigned int size);

int InterruptRead(int reader_index, int timeout);
void InterruptStop(int reader_index);
#endif
