/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 27, 2021.
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
/* 
 * Modification History
 *
 * November 1, 2001	Dieter Siegmund (dieter@apple.com)
 * - created
 */

#ifndef _EAP8021X_EAPCLIENTMODULE_H
#define _EAP8021X_EAPCLIENTMODULE_H

#include <stdint.h>

#include <EAP8021X/EAP.h>
#include <EAP8021X/EAPClientPlugin.h>

typedef struct EAPClientModule_s EAPClientModule, *EAPClientModuleRef;

enum {
    kEAPClientModuleStatusOK = 0,
    kEAPClientModuleStatusInvalidType = 1,
    kEAPClientModuleStatusTypeAlreadyLoaded = 2,
    kEAPClientModuleStatusAllocationFailed = 3,
    kEAPClientModuleStatusPluginInvalidVersion = 4,
    kEAPClientModuleStatusPluginIncomplete = 5,
};
typedef uint32_t EAPClientModuleStatus;

EAPClientModuleRef
EAPClientModuleLookup(EAPType type);

EAPType
EAPClientModuleDefaultType(void);

EAPClientModuleStatus
EAPClientModuleAddBuiltinModule(EAPClientPluginFuncIntrospect * func);

/*
 * Function: EAPClientModulePluginIntrospect
 * Returns:
 *   Given a function name, returns the corresponding function pointer
 *   by calling the plugin's "introspect" function, if supplied.
 *   The caller needs to know the prototype for the function i.e.
 *   what arguments to pass, and the return value.
 *   A module may or may not supply its introspect function for this
 *   purpose.
 */
EAPClientPluginFuncRef
EAPClientModulePluginIntrospect(EAPClientModuleRef module,
				EAPClientPluginFuncName);


EAPType
EAPClientModulePluginEAPType(EAPClientModuleRef module);

const char *
EAPClientModulePluginEAPName(EAPClientModuleRef module);

/*
 * EAPClientModulePlugin*
 * Functions to call the individual plug-in, given an EAPClientModule.
 * Note:
 *   These check for a NULL function pointer before calling the
 *   corresponding function.
 */

EAPClientStatus
EAPClientModulePluginInit(EAPClientModuleRef module, 
			  EAPClientPluginDataRef plugin, 
			  CFArrayRef * required_props,
			  int * error);

void 
EAPClientModulePluginFree(EAPClientModuleRef module,
			  EAPClientPluginDataRef plugin);

void 
EAPClientModulePluginFreePacket(EAPClientModuleRef module,
				EAPClientPluginDataRef plugin,
				EAPPacketRef pkt_p);
EAPClientState 
EAPClientModulePluginProcess(EAPClientModuleRef module,
			     EAPClientPluginDataRef plugin,
			     const EAPPacketRef in_pkt,
			     EAPPacketRef * out_pkt_p,
			     EAPClientStatus * status,
			     EAPClientDomainSpecificError * error);

const char * 
EAPClientModulePluginFailureString(EAPClientModuleRef module,
				   EAPClientPluginDataRef plugin);

void * 
EAPClientModulePluginSessionKey(EAPClientModuleRef module,
				EAPClientPluginDataRef plugin, 
				int * key_length);

void * 
EAPClientModulePluginServerKey(EAPClientModuleRef module,
			       EAPClientPluginDataRef plugin, 
			       int * key_length);

int
EAPClientModulePluginMasterSessionKeyCopyBytes(EAPClientModuleRef module,
					       EAPClientPluginDataRef plugin, 
					       uint8_t * msk, int msk_size);

CFArrayRef
EAPClientModulePluginRequireProperties(EAPClientModuleRef module,
				       EAPClientPluginDataRef plugin);

CFDictionaryRef
EAPClientModulePluginPublishProperties(EAPClientModuleRef module,
				       EAPClientPluginDataRef plugin);

bool
EAPClientModulePluginPacketDump(EAPClientModuleRef module,
				FILE * out_f, const EAPPacketRef packet);

CFStringRef
EAPClientModulePluginUserName(EAPClientModuleRef module,
			      CFDictionaryRef properties);

CFStringRef
EAPClientModulePluginCopyIdentity(EAPClientModuleRef module,
				  EAPClientPluginDataRef plugin);

CFStringRef
EAPClientModulePluginCopyPacketDescription(EAPClientModuleRef module,
					   const EAPPacketRef packet,
					   bool * is_valid);

#endif /* _EAP8021X_EAPCLIENTMODULE_H */
