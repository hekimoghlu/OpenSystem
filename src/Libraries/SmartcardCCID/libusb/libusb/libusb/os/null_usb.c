/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 28, 2022.
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
#include "libusbi.h"

static int
null_get_device_list(struct libusb_context * ctx,
	struct discovered_devs **discdevs)
{
	return LIBUSB_SUCCESS;
}

static int
null_open(struct libusb_device_handle *handle)
{
	return LIBUSB_ERROR_NOT_SUPPORTED;
}

static void
null_close(struct libusb_device_handle *handle)
{
}

static int
null_get_active_config_descriptor(struct libusb_device *dev,
    void *buf, size_t len)
{
	return LIBUSB_ERROR_NOT_SUPPORTED;
}

static int
null_get_config_descriptor(struct libusb_device *dev, uint8_t idx,
    void *buf, size_t len)
{
	return LIBUSB_ERROR_NOT_SUPPORTED;
}

static int
null_set_configuration(struct libusb_device_handle *handle, int config)
{
	return LIBUSB_ERROR_NOT_SUPPORTED;
}

static int
null_claim_interface(struct libusb_device_handle *handle, uint8_t iface)
{
	return LIBUSB_ERROR_NOT_SUPPORTED;
}

static int
null_release_interface(struct libusb_device_handle *handle, uint8_t iface)
{
	return LIBUSB_ERROR_NOT_SUPPORTED;
}

static int
null_set_interface_altsetting(struct libusb_device_handle *handle, uint8_t iface,
    uint8_t altsetting)
{
	return LIBUSB_ERROR_NOT_SUPPORTED;
}

static int
null_clear_halt(struct libusb_device_handle *handle, unsigned char endpoint)
{
	return LIBUSB_ERROR_NOT_SUPPORTED;
}

static int
null_submit_transfer(struct usbi_transfer *itransfer)
{
	return LIBUSB_ERROR_NOT_SUPPORTED;
}

static int
null_cancel_transfer(struct usbi_transfer *itransfer)
{
	return LIBUSB_ERROR_NOT_SUPPORTED;
}

const struct usbi_os_backend usbi_backend = {
	.name = "Null backend",
	.caps = 0,
	.get_device_list = null_get_device_list,
	.open = null_open,
	.close = null_close,
	.get_active_config_descriptor = null_get_active_config_descriptor,
	.get_config_descriptor = null_get_config_descriptor,
	.set_configuration = null_set_configuration,
	.claim_interface = null_claim_interface,
	.release_interface = null_release_interface,
	.set_interface_altsetting = null_set_interface_altsetting,
	.clear_halt = null_clear_halt,
	.submit_transfer = null_submit_transfer,
	.cancel_transfer = null_cancel_transfer,
};
