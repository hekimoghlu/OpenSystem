/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 2, 2021.
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
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>
#include <libusb.h>
#include <errno.h>

#include "config.h"

#define KOBIL_VENDOR_ID		0x0D46
#define MID_DEVICE_ID		0x4081
#define KOBIL_TIMEOUT		5000
#define VAL_STARTUP_4080        1
#define VAL_STARTUP_4000        2
#define VAL_STARTUP_4020        3
#define VAL_STARTUP_40A0        4
#define HIDCMD_SWITCH_DEVICE    0x0004

#define bmRequestType           0x22
#define bRequest                0x09
#define wValue                  0x0200
#define wIndex                  0x0002  /* this was originally 0x0001 */


static int kobil_midentity_control_msg(libusb_device_handle *usb)
{
	int ret;

	unsigned char switchCmd[10];

	unsigned char Sleep = 1;
	unsigned char hardDisk = 1;

	unsigned char param = ((hardDisk) << 4) | (Sleep);

	memset(switchCmd, 0x0, sizeof(switchCmd));
	switchCmd[0] = HIDCMD_SWITCH_DEVICE >> 8;
	switchCmd[1] = HIDCMD_SWITCH_DEVICE;
	switchCmd[5] = VAL_STARTUP_4000;
	switchCmd[9] = param;

	ret = libusb_control_transfer(usb, bmRequestType, bRequest, wValue, wIndex,
			switchCmd, sizeof(switchCmd), KOBIL_TIMEOUT);

	return(!(ret==sizeof(switchCmd)));
}


static int kobil_midentity_claim_interface(libusb_device_handle *usb, int ifnum)
{
	int rv;

	printf("claiming interface #%d ... ", ifnum);
	rv = libusb_claim_interface(usb, ifnum);
	if (rv == 0)
	{
		printf("success\n");
		return rv;
	}
	else
		printf("failed\n");

	printf("failed with error %d, trying to detach kernel driver ....\n", rv);
	rv = libusb_detach_kernel_driver(usb, ifnum);
	if (rv == 0)
	{
		printf("success, claiming interface again ...");
		rv = libusb_claim_interface(usb, ifnum);
		if (rv == 0)
		{
			printf("success\n");
			return rv;
		}
		else
			printf("failed\n");
	}

	printf("failed with error %d, giving up.\n", rv);
	return rv;
}


int main(int argc, char *argv[])
{
	libusb_device **devs, *dev;
	libusb_device *found_dev = NULL;
	struct libusb_device_handle *usb = NULL;
	int rv, i;
	ssize_t cnt;

	(void)argc;
	(void)argv;

	rv = libusb_init(NULL);
	if (rv < 0)
	{
		(void)printf("libusb_init() failed\n");
		return rv;
	}

	cnt = libusb_get_device_list(NULL, &devs);
	if (cnt < 0)
	{
		(void)printf("libusb_get_device_list() failed\n");
		return (int)cnt;
	}

	/* for every device */
	i = 0;
	while ((dev = devs[i++]) != NULL)
	{
		struct libusb_device_descriptor desc;

		rv = libusb_get_device_descriptor(dev, &desc);
		if (rv < 0) {
			(void)printf("failed to get device descriptor\n");
			continue;
		}

		printf("vendor/product: %04X %04X\n", desc.idVendor, desc.idProduct);
		if (desc.idVendor == KOBIL_VENDOR_ID && desc.idProduct == MID_DEVICE_ID)
			found_dev = dev;
	}

	if (found_dev == NULL)
	{
		printf("device not found. aborting.\n");
		if (0 != geteuid())
			printf("Try to rerun this program as root.\n");
		exit(1);
	}

	printf("Device found, opening ... ");
	rv = libusb_open(found_dev, &usb);
	if (rv < 0)
	{
		printf("failed, aborting.\n");
		exit(2);
	}
	printf("success\n");

	rv = kobil_midentity_claim_interface(usb, 0);
	if (rv < 0)
	{
		libusb_close(usb);
		exit(3);
	}

	rv = kobil_midentity_claim_interface(usb, 1);
	if (rv < 0)
	{
		libusb_close(usb);
		exit(3);
	}

	printf("Activating the CCID configuration .... ");
	rv = kobil_midentity_control_msg(usb);
	if (rv == 0)
		printf("success\n");
	else
		printf("failed with error %d, giving up.\n", rv);

	libusb_close(usb);

	return 0;
}

