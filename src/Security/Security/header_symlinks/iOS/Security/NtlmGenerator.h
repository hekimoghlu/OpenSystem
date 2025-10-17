/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 16, 2024.
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
#ifndef _NTLM_GENERATOR_H_
#define _NTLM_GENERATOR_H_

#include <CoreFoundation/CFData.h>
#include <CoreFoundation/CFString.h>

#ifdef  __cplusplus
extern "C" {
#endif

/*
 * This interface provides the capability to generate and parse the authentication
 * blobs which pass back and forth between a client and a server during NTLM
 * authentication. Only the client side is implemented. 
 *
 * All three variants of NTLM authentication are performed: NTLM1, NTLM2, and 
 * NTLMv2. 
 *
 * In general, to use this stuff for HTTP authentication:
 *
 * 1. Determine that NTLM authentication is possible. Drop the connection
 *    to the server if you have a persistent connection open; MS servers
 *    require a clean unused connection for this negotiation to occur. 
 *
 * 2. Create a NtlmGeneratorRef object, specifying possible restrictions
 *    on negotiation version. 
 *
 * 3. Create the client authentication blob using NtlmCreateClientRequest()
 *    and send it to the server, base64 encoded, in a "Authorization: NTLM" 
 *    header line. 
 *
 * 4. The server should send back another 401 status, with its own blob in
 *    a "WWW-Authenticate: NTLM" header. 
 *
 * 5. Base64 decode that blob and feed it into NtlmCreateClientResponse(), the 
 *    output of which is another blob which you send to the server again in 
 *    a "WWW-Authenticate: NTLM" header. 
 *
 * 6. If you're lucky the server will give a 200 status (or something else useful
 *    other than 401) and you're done. 
 *
 * 7. Free the NtlmGeneratorRef object with NtlmGeneratorRelease().
 */
 
/*
 * Opaque reference to an NTLM blob generator object.
 */
typedef struct NtlmGenerator *NtlmGeneratorRef;

/*
 * Which versions of the protocol are acceptable?
 */
enum {
	NW_NTLM1   = 0x00000001,
	NW_NTLM2   = 0x00000002,
	NW_NTLMv2  = 0x00000004,

	// all variants enabled, preferring NTLMv2, then NTLM2
	NW_Any     = NW_NTLM2 | NW_NTLMv2
};
typedef uint32_t NLTM_Which;


/* Create/release NtlmGenerator objects.*/
OSStatus NtlmGeneratorCreate(
	NLTM_Which			which,
	NtlmGeneratorRef	*ntlmGen);			/* RETURNED */
	
void NtlmGeneratorRelease(
	NtlmGeneratorRef	ntlmGen);
	
/* create the initial client request */
OSStatus NtlmCreateClientRequest(
	NtlmGeneratorRef	ntlmGen,
	CFDataRef			*clientRequest);	/* RETURNED */
		
/* parse server challenge and respond to it */
OSStatus NtlmCreateClientResponse(
	NtlmGeneratorRef	ntlmGen,
	CFDataRef			serverBlob,			/* obtained from the server */
	CFStringRef			domain,				/* server domain, appears to be optional */
	CFStringRef			userName,
	CFStringRef			password,
	CFDataRef			*clientResponse);   /* RETURNED */
		
/* which version did we negotiate? */
NLTM_Which NtlmGetNegotiatedVersion(
	NtlmGeneratorRef	ntlmGen);

OSStatus NtlmGeneratePasswordHashes(
	CFAllocatorRef alloc,
	CFStringRef password,
	CFDataRef* ntlmHash,
	CFDataRef* lmHash);

OSStatus _NtlmCreateClientResponse(
	NtlmGeneratorRef	ntlmGen,
	CFDataRef			serverBlob,
	CFStringRef			domain,				/* optional */
	CFStringRef			userName,
	CFDataRef			ntlmHash,
	CFDataRef			lmHash,
	CFDataRef			*clientResponse);	/* RETURNED */
																			
#ifdef  __cplusplus
}
#endif

#endif  /* _NTLM_GENERATOR_H_ */
