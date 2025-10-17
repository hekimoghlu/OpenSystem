/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 8, 2023.
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
#ifndef _KEXT_AUDIT_USER_CLIENT_H_
#define _KEXT_AUDIT_USER_CLIENT_H_

#include <IOKit/IOLib.h>
#include <IOKit/IOUserClient.h>

#include "KextAudit.h"

#define VALID_KEXT_LOADTYPE(loadType)		\
	((loadType) == kKALTKextCDHashSha1 ||	\
	 (loadType) == kKALTKextCDHashSha256)

// <rdar://problem/30172421> Expose t208/t290 defines into macOS SDK
enum KextAuditBridgeDeviceType {
	kKextAuditBridgeDeviceTypeNoCoprocessor = 0x00000000,
	kKextAuditBridgeDeviceTypeT208 = 0x00010000,
	kKextAuditBridgeDeviceTypeT290 = 0x00020000,
};

class KextAuditUserClient : public IOUserClient
{
	OSDeclareDefaultStructors(KextAuditUserClient);

protected:
	static const IOExternalMethodDispatch sMethods[kKextAuditMethodCount];
	static IOReturn notifyLoad(KextAuditUserClient *target, void *, IOExternalMethodArguments *args);
	static IOReturn test(KextAuditUserClient *target, void *, IOExternalMethodArguments *args);

private:
	task_t fTask;
	KextAudit *fProvider;
	bool fUserClientHasEntitlement;
	KextAuditBridgeDeviceType fDeviceType;

public:
	virtual IOReturn clientClose(void) override;
	virtual bool initWithTask(task_t owningTask, void *security_id, UInt32 type,
	                          OSDictionary *properties) override;
	virtual bool start(IOService *provider) override;
	virtual void stop(IOService *provider) override;
	virtual void free(void) override;
	IOReturn externalMethod(uint32_t selector, IOExternalMethodArguments *arguments,
	                        IOExternalMethodDispatch *dispatch, OSObject *target,
	                        void *reference) override;
	KextAuditBridgeDeviceType getBridgeDeviceType(void);
};

#endif /* !_KEXT_AUDIT_USER_CLIENT_H_ */
