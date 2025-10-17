/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 6, 2025.
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
#include <sys/cdefs.h>
#include <sys/types.h>
#include <stdint.h>

/*
	File:		pbkdf2.h
	Contains:	Apple Data Security Services PKCS #5 PBKDF2 function declaration.
	Copyright (c) 1999,2012,2014 Apple Inc. All Rights Reserved.
*/

#ifndef __PBKDF2__
#define __PBKDF2__

__BEGIN_DECLS

/* This function should generate a pseudo random octect stream
   of hLen bytes long (The value hLen is specified as an argument to pbkdf2
   and should be constant for any given prf function.) which is output in the buffer
   pointed to by randomPtr (the caller of this function is responsible for allocation
   of the buffer).
   The inputs to the pseudo random function are the first keyLen octets pointed
   to by keyPtr and the first textLen octets pointed to by textPtr.
   Both keyLen and textLen can have any nonzero value.
   A good prf would be a HMAC-SHA-1 algorithm where the keyPtr octets serve as
   HMAC's "key" and the textPtr octets serve as HMAC's "text".  */
typedef void (*PRF)(const uint8_t *keyPtr, size_t keyLen,
					const uint8_t *textPtr, size_t textLen,
					uint8_t *randomPtr);

/* This function implements the PBKDF2 key derrivation algorithm described in
   http://www.rsa.com/rsalabs/pubs/PKCS/html/pkcs-5.html
   The output is a derived key of dkLen bytes which is written to the buffer
   pointed to by dkPtr.
   The caller should ensure dkPtr is at least dkLen bytes long.
   The Key is derived from passwordPtr (which is passwordLen bytes long) and from
   saltPtr (which is saltLen bytes long).  The algorithm used is desacribed in
   PKCS #5 version 2.0 and iterationCount iterations are performed.
   The argument prf is a pointer to a psuedo random number generator declared above.
   It should write exactly hLen bytes into its output buffer each time it is called.
   The argument tempBuffer should point to a buffer MAX (hLen, saltLen + 4) + 2 * hLen
   bytes long.  This buffer is used during the calculation for intermediate results.
   Security Considerations:
   The argument saltPtr should be a pointer to a buffer of at least 8 random bytes
   (64 bits).  Thus saltLen should be >= 8.
   For each session a new salt should be generated.
   The value of iterationCount should be at least 1000 (one thousand).
   A good prf would be a HMAC-SHA-1 algorithm where the password serves as
   HMAC's "key" and the data serves as HMAC's "text".  */
void pbkdf2 (PRF prf, size_t hLen,
			 const void *passwordPtr, size_t passwordLen,
			 const void *saltPtr, size_t saltLen,
			 size_t iterationCount,
			 void *dkPtr, size_t dkLen,
			 void *tempBuffer);


#ifdef	__cplusplus
}
#endif

#endif /* __PBKDF2__ */
