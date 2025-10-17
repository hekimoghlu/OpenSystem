/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 7, 2022.
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
#include <List.h>
#include <Locker.h>
#include <Autolock.h>
#include <USBKit.h>
#include <map>
#include "libusbi.h"
#include "haiku_usb_raw.h"

using namespace std;

class USBDevice;
class USBDeviceHandle;
class USBTransfer;

class USBDevice {
public:
						USBDevice(const char *);
	virtual					~USBDevice();
	const char*				Location() const;
	uint8					CountConfigurations() const;
	const usb_device_descriptor*		Descriptor() const;
	const usb_configuration_descriptor*	ConfigurationDescriptor(uint8) const;
	const usb_configuration_descriptor*	ActiveConfiguration() const;
	uint8					EndpointToIndex(uint8) const;
	uint8					EndpointToInterface(uint8) const;
	int					ClaimInterface(uint8);
	int					ReleaseInterface(uint8);
	int					CheckInterfacesFree(uint8);
	void					SetActiveConfiguration(uint8);
	uint8					ActiveConfigurationIndex() const;
	bool					InitCheck();
private:
	int					Initialise();
	unsigned int				fClaimedInterfaces;	// Max Interfaces can be 32. Using a bitmask
	usb_device_descriptor			fDeviceDescriptor;
	unsigned char**				fConfigurationDescriptors;
	uint8					fActiveConfiguration;
	char*					fPath;
	map<uint8,uint8>			fConfigToIndex;
	map<uint8,uint8>*			fEndpointToIndex;
	map<uint8,uint8>*			fEndpointToInterface;
	bool					fInitCheck;
};

class USBDeviceHandle {
public:
				USBDeviceHandle(USBDevice *dev);
	virtual			~USBDeviceHandle();
	int			ClaimInterface(uint8);
	int			ReleaseInterface(uint8);
	int			SetConfiguration(uint8);
	int			SetAltSetting(uint8, uint8);
	int			ClearHalt(uint8);
	status_t		SubmitTransfer(struct usbi_transfer *);
	status_t		CancelTransfer(USBTransfer *);
	bool			InitCheck();
private:
	int			fRawFD;
	static status_t		TransfersThread(void *);
	void			TransfersWorker();
	USBDevice*		fUSBDevice;
	unsigned int		fClaimedInterfaces;
	BList			fTransfers;
	BLocker			fTransfersLock;
	sem_id			fTransfersSem;
	thread_id		fTransfersThread;
	bool			fInitCheck;
};

class USBTransfer {
public:
					USBTransfer(struct usbi_transfer *, USBDevice *);
	virtual				~USBTransfer();
	void				Do(int);
	struct usbi_transfer*		UsbiTransfer();
	void				SetCancelled();
	bool				IsCancelled();
private:
	struct usbi_transfer*		fUsbiTransfer;
	struct libusb_transfer*		fLibusbTransfer;
	USBDevice*			fUSBDevice;
	BLocker				fStatusLock;
	bool				fCancelled;
};

class USBRoster {
public:
			USBRoster();
	virtual		~USBRoster();
	int		Start();
	void		Stop();
private:
	void*		fLooper;
};
