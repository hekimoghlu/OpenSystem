/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 12, 2021.
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
 * DH_csp.cpp - Diffie-Hellman Algorithm factory
 */
 
#include "DH_csp.h"
#include "DH_keys.h"
#include <Security/cssmapple.h>

Allocator *DH_Factory::normAllocator;
Allocator *DH_Factory::privAllocator;

DH_Factory::DH_Factory(Allocator *normAlloc, Allocator *privAlloc)
{
	setNormAllocator(normAlloc);
	setPrivAllocator(privAlloc);
	
	/* NOTE WELL we assume that the RSA_DSA factory has already been instantitated, 
	 * doing the basic init of openssl */
	 
	ERR_load_DH_strings();
}

DH_Factory::~DH_Factory()
{
}

bool DH_Factory::setup(
	AppleCSPSession &session,	
	CSPFullPluginSession::CSPContext * &cspCtx, 
	const Context &context)
{
	switch(context.type()) {
		case CSSM_ALGCLASS_KEYGEN:
			switch(context.algorithm()) {
				case CSSM_ALGID_DH:
					if(cspCtx == NULL) {
						cspCtx = new DHKeyPairGenContext(session, context);
					}
					return true;
				default:
					break;
			}
			break;		

		default:
			break;
	}
	/* not implemented here */
	return false;
}



