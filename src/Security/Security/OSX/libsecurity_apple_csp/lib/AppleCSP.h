/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 30, 2022.
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
//
// AppleCSP.h - top-level plugin and session classes
//
#ifndef _APPLE_CSP_H_
#define _APPLE_CSP_H_

#include <security_cdsa_plugin/cssmplugin.h>
#include <security_cdsa_plugin/pluginsession.h>
#include <security_cdsa_plugin/CSPsession.h>

class AppleCSPSession;
class AppleCSPContext; 

/*
 * AppleCSP-specific algorithm factory. 
 */
class AppleCSPAlgorithmFactory {
public:
	AppleCSPAlgorithmFactory() {};
	virtual ~AppleCSPAlgorithmFactory() { };

	// set ctx and return true if you can handle this
	virtual bool setup(
		AppleCSPSession 					&session,
		CSPFullPluginSession::CSPContext 	* &cspCtx, 
		const Context &context) = 0;
		
	/* probably other setup methods, e.g. by CSSM_ALGORITHMS instead of 
	 * context */
};

class AppleCSPPlugin : public CssmPlugin {
    friend class AppleCSPSession;
	friend class AppleCSPContext;
	
public:
    AppleCSPPlugin();
    ~AppleCSPPlugin();

    PluginSession *makeSession(CSSM_MODULE_HANDLE handle,
                               const CSSM_VERSION &version,
                               uint32 subserviceId,
                               CSSM_SERVICE_TYPE subserviceType,
                               CSSM_ATTACH_FLAGS attachFlags,
                               const CSSM_UPCALLS &upcalls);

	Allocator 	&normAlloc()	{return normAllocator; }
    Allocator 	&privAlloc()	{return privAllocator; }

private:
    Allocator 				&normAllocator;	
    Allocator 				&privAllocator;	
	#ifdef	CRYPTKIT_CSP_ENABLE
	AppleCSPAlgorithmFactory	*cryptKitFactory;		
	#endif
	AppleCSPAlgorithmFactory	*miscAlgFactory;
	#ifdef	ASC_CSP_ENABLE
	AppleCSPAlgorithmFactory	*ascAlgFactory;
	#endif
	AppleCSPAlgorithmFactory	*rsaDsaAlgFactory;
	AppleCSPAlgorithmFactory	*dhAlgFactory;
};


#endif //_APPLE_CSP_H_
