/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 29, 2022.
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


#import <CoreFoundation/CoreFoundation.h>
#import <Security/Security.h>

typedef struct NAHData *NAHRef;
typedef struct NAHSelectionData *NAHSelectionRef;

/*
 * Flow:
 *
 * netauth = NAHCreate(CFSTR("hostname"), CFSTR("service"),
 *             { username = "foo", supportedMechs = mechs } );
 *
 * selections = NAHGetSelections(netauth);
 *
 * foreach s (selections) { 
 *
 *    acquiredict = [[NSMutableDictionary alloc] init];
 *    [acquiredict setValue:@"password" forKey:kNAHPassword];
 *
 *    if !(NAHSelectionAcquireCredential(s, acquiredict, NULL))  #blocking
 *          continue;
 *
 *    dict = NAHSelectionCopyAuthInfo(s)
 *
 *    res = NetFSOpenSesssion(dict);
 *    if (res == sucess) {
 *        CFRelese(dict);
 *        break;
 *    } else if (sucess == authentication_failed) {
 *        
 *    } else {
 *         #ignore all other failures
 *    }
 *    CFRelese(dict);
 * }
 *
 * CFRelese(netauth);
 * 
 */

extern const CFStringRef kNAHErrorDomain;

/* service keys */

extern const CFStringRef kNAHServiceAFPServer;
extern const CFStringRef kNAHServiceCIFSServer;
extern const CFStringRef kNAHServiceHostServer;
extern const CFStringRef kNAHServiceVNCServer;


extern const CFStringRef kNAHNegTokenInit; /* private - CFDictRef */
extern const CFStringRef kNAHUserName; /* CFStringRef */

/*
 * Default is to consider all certficates accessable in key chain.
 *
 * If this key is used, only these will only considered, pass in a
 * empty CFArrayRef if you want to disable using certificates.
 */

extern const CFStringRef kNAHCertificates; /* SecIdentityRef/CFArrayRef */
extern const CFStringRef kNAHPassword;

NAHRef
NAHCreate(CFAllocatorRef alloc,
	 CFStringRef hostname,
	 CFStringRef service,
	 CFDictionaryRef info);

/*
 * Return a ordered list of authentication
 */

CFArrayRef
NAHGetSelections(NAHRef);

extern const CFStringRef kNAHForceRefreshCredential;

Boolean
NAHSelectionAcquireCredential(NAHSelectionRef selection,
			     CFDictionaryRef info,
			     CFErrorRef *error);

Boolean
NAHSelectionAcquireCredentialAsync(NAHSelectionRef selection,
				  CFDictionaryRef info,
				  dispatch_queue_t q,
				  void (^result)(CFErrorRef error));

void
NAHCancel(NAHRef na);

/*
 * Status of selection
 */

extern const CFStringRef kNAHSelectionHaveCredential;
extern const CFStringRef kNAHSelectionUserPrintable;
extern const CFStringRef kNAHClientPrincipal;
extern const CFStringRef kNAHServerPrincipal;
extern const CFStringRef kNAHMechanism;
extern const CFStringRef kNAHInnerMechanism;
extern const CFStringRef kNAHCredentialType;
extern const CFStringRef kNAHUseSPNEGO;

extern const CFStringRef kNAHClientNameType;
extern const CFStringRef kNAHClientNameTypeGSSD;

extern const CFStringRef kNAHServerNameType;
extern const CFStringRef kNAHServerNameTypeGSSD;

extern const CFStringRef kNAHInferredLabel;

extern const CFStringRef kNAHNTUsername;
extern const CFStringRef kNAHNTServiceBasedName;
extern const CFStringRef kNAHNTKRB5PrincipalReferral;
extern const CFStringRef kNAHNTKRB5Principal;
extern const CFStringRef kNAHNTUUID;


CFTypeRef
NAHSelectionGetInfoForKey(NAHSelectionRef selection, CFStringRef key);

CFDictionaryRef
NAHSelectionCopyAuthInfo(NAHSelectionRef selection);

/*
 * Reference counting
 */

CFStringRef
NAHCopyReferenceKey(NAHSelectionRef selection);

Boolean
NAHAddReferenceAndLabel(NAHSelectionRef client, CFStringRef identifier);
		  
void
NAHFindByLabelAndRelease(CFStringRef identifier);

Boolean
NAHCredAddReference(CFStringRef referenceKey);

Boolean
NAHCredRemoveReference(CFStringRef referenceKey);

char *
NAHCreateRefLabelFromIdentifier(CFStringRef identifier);

extern CFStringRef kGSSAPIMechNTLM;
extern CFStringRef kGSSAPIMechKerberos;
extern CFStringRef kGSSAPIMechKerberosU2U;
extern CFStringRef kGSSAPIMechKerberosMicrosoft;
extern CFStringRef kGSSAPIMechPKU2U;
extern CFStringRef kGSSAPIMechIAKerb;
extern CFStringRef kGSSAPIMechSPNEGO;

CFStringRef
NAHCopyMMeUserNameFromCertificate(SecCertificateRef cert);
