/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 17, 2022.
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
#ifndef _KEXT_AUDIT_H_
#define _KEXT_AUDIT_H_

#include <stdint.h>
#include <IOKit/IOLib.h>
#include <IOKit/IOLocks.h>
#include <IOKit/IOService.h>
#include <IOKit/smc/AppleSMCFamily.h>

#include <MultiverseSupport/kext_audit_plugin_common.h>
#include "efi_smc.h"

#ifdef DEBUG
#define DEBUG_LOG(fmt, ...) IOLog("%s, in %s, line %d: " fmt "\n", "KextAudit",\
				__func__, __LINE__, ##__VA_ARGS__)
#else
#define DEBUG_LOG(fmt, ...)
#endif /* DEBUG */

#define kKextAuditPollIntervalMs  2

#define kKextAuditUserAccessEntitlement "com.apple.private.KextAudit.user-access"

class KextAudit : public IOService
{
	OSDeclareDefaultStructors(KextAudit)

public:
	virtual bool init(OSDictionary *dictionary) override;
	virtual void free(void) override;
	virtual IOService *probe(IOService *provider, SInt32 *score) override;
	virtual bool start(IOService *provider) override;
	virtual void stop(IOService *provider) override;
	virtual bool terminate(IOOptionBits options) override;

	bool notifyBridgeWithReplySync(struct KextAuditLoadNotificationKext *kaln,
	                               struct KextAuditBridgeResponse *kabr);

	bool testBridgeConnection(struct KextAuditBridgeResponse *kabr);

private:
	AppleSMCFamily *fSMCDriver;
	IOLock *_kalnLock;
};

#endif /* _KEXT_AUDIT_H_ */
