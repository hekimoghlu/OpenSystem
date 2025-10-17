/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 6, 2022.
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
 * RSA_DSA_csp.h - Algorithm factory for RSA/DSA
 */
 
#ifndef	_RSA_DSA_CSP_H_
#define _RSA_DSA_CSP_H_

#include <security_cdsa_plugin/CSPsession.h>
#include <AppleCSP.h>

/* Can't include AppleCSPSession.h due to circular dependency */
class AppleCSPSession;

class RSA_DSA_Factory : public AppleCSPAlgorithmFactory {
public:
    RSA_DSA_Factory(Allocator *normAlloc = NULL, Allocator *privAlloc = NULL);
	~RSA_DSA_Factory();
	
    bool setup(
		AppleCSPSession &session,
		CSPFullPluginSession::CSPContext * &cspCtx, 
		const Context &context);

    static void setNormAllocator(Allocator *alloc)
    { assert(!normAllocator); normAllocator = alloc; }
    static void setPrivAllocator(Allocator *alloc)
    { assert(!privAllocator); privAllocator = alloc; }

    // memory allocators
    static Allocator *normAllocator;
    static Allocator *privAllocator;
    
};

#endif	/* _RSA_DSA_CSP_H_ */
