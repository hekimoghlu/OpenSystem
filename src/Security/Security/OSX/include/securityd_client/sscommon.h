/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 13, 2022.
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
// sscommon - common definitions for all securityd MIG interfaces
//
// This is meant to go into both ssclient and tdclient (for tokend), so it
// needs to be fairly generic.
//
#ifndef _H_SSCOMMON
#define _H_SSCOMMON

#include <Security/cssm.h>

//
// some handle types used to be defined here, so don't break anybody still 
// relying on that
//
#include <securityd_client/handletypes.h>

#ifdef __cplusplus

#include <security_utilities/alloc.h>
#include <security_utilities/mach++.h>
#include <security_cdsa_utilities/cssmdata.h>
#include <security_cdsa_utilities/cssmkey.h>
#include <security_cdsa_utilities/cssmacl.h>
#include <security_cdsa_utilities/context.h>
#include <security_cdsa_utilities/cssmdbname.h>
#include <security_cdsa_utilities/cssmdb.h>


namespace Security {
namespace SecurityServer {

using MachPlusPlus::Port;
using MachPlusPlus::ReceivePort;
using MachPlusPlus::ReplyPort;

#endif //__cplusplus


//
// The Mach bootstrap registration name for SecurityServer
//
#define SECURITYSERVER_BOOTSTRAP_NAME	"com.apple.SecurityServer"

//
// Types of ACL bearers
//
typedef enum { dbAcl, keyAcl, objectAcl, loginAcl } AclKind;


#ifdef __cplusplus


//
// Common structure for IPC-client mediator objects
//
class ClientCommon {
	NOCOPY(ClientCommon)
public:
	ClientCommon(Allocator &standard = Allocator::standard(),
		Allocator &returning = Allocator::standard())
		: internalAllocator(standard), returnAllocator(returning) { }

	Allocator &internalAllocator;
	Allocator &returnAllocator;

public:
	typedef Security::Context Context;
};



} // end namespace SecurityServer
} // end namespace Security

#endif //__cplusplus


#endif //_H_SSCOMMON
