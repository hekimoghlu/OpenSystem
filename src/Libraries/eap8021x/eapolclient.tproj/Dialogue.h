/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 4, 2024.
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
 * November 8, 2001	Dieter Siegmund
 * - created
 */
 
#ifndef _S_DIALOGUE_H
#define _S_DIALOGUE_H

#include <sys/types.h>
#include <CoreFoundation/CFString.h>
#include <Security/SecIdentity.h>

/**
 ** CredentialsDialogue
 **/
typedef struct {
    CFStringRef		username;
    CFStringRef		password;
    CFStringRef		new_password;
    Boolean		user_cancelled;
    Boolean		remember_information;
    SecIdentityRef	chosen_identity;
} CredentialsDialogueResponse, *CredentialsDialogueResponseRef;

typedef void 
(*CredentialsDialogueResponseCallBack)(const void * arg1, 
				       const void * arg2, 
				       CredentialsDialogueResponseRef data);

typedef struct CredentialsDialogue_s CredentialsDialogue, 
    *CredentialsDialogueRef;

extern const CFStringRef	kCredentialsDialogueSSID;
extern const CFStringRef	kCredentialsDialogueAccountName;
extern const CFStringRef	kCredentialsDialoguePassword;
extern const CFStringRef	kCredentialsDialogueCertificates;
extern const CFStringRef	kCredentialsDialogueRememberInformation;
extern const CFStringRef	kCredentialsDialoguePasswordChangeRequired;

CredentialsDialogueRef
CredentialsDialogue_create(CredentialsDialogueResponseCallBack func,
			   const void * arg1, const void * arg2, 
			   CFDictionaryRef details);
void
CredentialsDialogue_free(CredentialsDialogueRef * dialogue_p_p);

/**
 ** TrustDialogue
 **/
typedef struct {
    Boolean		proceed;
} TrustDialogueResponse, *TrustDialogueResponseRef;

typedef void 
(*TrustDialogueResponseCallBack)(const void * arg1, 
				 const void * arg2, 
				 TrustDialogueResponseRef data);

typedef struct TrustDialogue_s TrustDialogue, *TrustDialogueRef;

TrustDialogueRef
TrustDialogue_create(TrustDialogueResponseCallBack func,
		     const void * arg1, const void * arg2,
		     CFDictionaryRef trust_info,
		     CFTypeRef ssid, CFStringRef interface);

CFDictionaryRef
TrustDialogue_trust_info(TrustDialogueRef dialogue);

void
TrustDialogue_free(TrustDialogueRef * dialogue_p_p);

/**
 ** AlertDialogue
 **/

typedef void 
(*AlertDialogueResponseCallBack)(const void * arg1, 
				 const void * arg2);

typedef struct AlertDialogue_s AlertDialogue, *AlertDialogueRef;

AlertDialogueRef
AlertDialogue_create(AlertDialogueResponseCallBack func,
		     const void * arg1, const void * arg2,
		     CFStringRef message, CFStringRef ssid);
void
AlertDialogue_free(AlertDialogueRef * dialogue_p_p);


#endif /* _S_DIALOGUE_H */
