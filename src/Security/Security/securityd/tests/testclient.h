/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 12, 2025.
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
// Tester - test driver for securityserver client side.
//
#ifndef _H_TESTCLIENT
#define _H_TESTCLIENT

#include "ssclient.h"
#include <Security/cssmerrno.h>
#include <Security/debugging.h>
#include <Security/cssmclient.h>
#include <Security/signclient.h>
#include <Security/cryptoclient.h>
#include <stdarg.h>


//
// Names from the SecurityServerSession class
//
using namespace SecurityServer;
using namespace CssmClient;


//
// Test drivers
//
void integrity();
void signWithRSA();
void desEncryption();
void blobs();
void keyBlobs();
void databases();
void timeouts();
void acls();
void authAcls();
void codeSigning();
void keychainAcls();
void authorizations();
void adhoc();


//
// Global constants
//
extern const CssmData null;					// zero pointer, zero length constant data
extern const AccessCredentials nullCred;	// null credentials

extern CSSM_GUID ssguid;					// a fixed guid
extern CssmSubserviceUid ssuid;				// a subservice-uid using this guid


#endif //_H_TESTCLIENT
