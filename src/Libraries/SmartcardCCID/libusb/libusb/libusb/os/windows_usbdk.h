/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 18, 2022.
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
#ifndef LIBUSB_WINDOWS_USBDK_H
#define LIBUSB_WINDOWS_USBDK_H

#include "windows_common.h"

typedef struct USB_DK_CONFIG_DESCRIPTOR_REQUEST {
	USB_DK_DEVICE_ID ID;
	ULONG64 Index;
} USB_DK_CONFIG_DESCRIPTOR_REQUEST, *PUSB_DK_CONFIG_DESCRIPTOR_REQUEST;

typedef enum {
	TransferFailure = 0,
	TransferSuccess,
	TransferSuccessAsync
} TransferResult;

typedef enum {
	NoSpeed = 0,
	LowSpeed,
	FullSpeed,
	HighSpeed,
	SuperSpeed
} USB_DK_DEVICE_SPEED;

typedef enum {
	ControlTransferType,
	BulkTransferType,
	InterruptTransferType,
	IsochronousTransferType
} USB_DK_TRANSFER_TYPE;

typedef BOOL (__cdecl *USBDK_GET_DEVICES_LIST)(
	PUSB_DK_DEVICE_INFO *DeviceInfo,
	PULONG DeviceNumber
);
typedef void (__cdecl *USBDK_RELEASE_DEVICES_LIST)(
	PUSB_DK_DEVICE_INFO DeviceInfo
);
typedef HANDLE (__cdecl *USBDK_START_REDIRECT)(
	PUSB_DK_DEVICE_ID DeviceId
);
typedef BOOL (__cdecl *USBDK_STOP_REDIRECT)(
	HANDLE DeviceHandle
);
typedef BOOL (__cdecl *USBDK_GET_CONFIGURATION_DESCRIPTOR)(
	PUSB_DK_CONFIG_DESCRIPTOR_REQUEST Request,
	PUSB_CONFIGURATION_DESCRIPTOR *Descriptor,
	PULONG Length
);
typedef void (__cdecl *USBDK_RELEASE_CONFIGURATION_DESCRIPTOR)(
	PUSB_CONFIGURATION_DESCRIPTOR Descriptor
);
typedef TransferResult (__cdecl *USBDK_WRITE_PIPE)(
	HANDLE DeviceHandle,
	PUSB_DK_TRANSFER_REQUEST Request,
	LPOVERLAPPED lpOverlapped
);
typedef TransferResult (__cdecl *USBDK_READ_PIPE)(
	HANDLE DeviceHandle,
	PUSB_DK_TRANSFER_REQUEST Request,
	LPOVERLAPPED lpOverlapped
);
typedef BOOL (__cdecl *USBDK_ABORT_PIPE)(
	HANDLE DeviceHandle,
	ULONG64 PipeAddress
);
typedef BOOL (__cdecl *USBDK_RESET_PIPE)(
	HANDLE DeviceHandle,
	ULONG64 PipeAddress
);
typedef BOOL (__cdecl *USBDK_SET_ALTSETTING)(
	HANDLE DeviceHandle,
	ULONG64 InterfaceIdx,
	ULONG64 AltSettingIdx
);
typedef BOOL (__cdecl *USBDK_RESET_DEVICE)(
	HANDLE DeviceHandle
);
typedef HANDLE (__cdecl *USBDK_GET_REDIRECTOR_SYSTEM_HANDLE)(
	HANDLE DeviceHandle
);

#endif
