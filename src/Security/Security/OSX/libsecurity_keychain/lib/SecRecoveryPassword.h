/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 25, 2022.
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
 * generalize to Sec function
 */

#ifndef SEC_RECOVERYPASSWORD_H
#define SEC_RECOVERYPASSWORD_H

#include <stdint.h>
#include <stddef.h>
#include <CoreFoundation/CoreFoundation.h>
#include <Availability.h>

#ifdef __cplusplus
extern "C" {
#endif

extern CFStringRef kSecRecVersionNumber,
				   kSecRecQuestions,
				   kSecRecLocale,
				   kSecRecIV,
				   kSecRecWrappedPassword;

/*!
     @function	SecWrapRecoveryPasswordWithAnswers
     @abstract	Wrap a password with a key derived from an array of answers to questions 
     
     @param		password	The password to wrap.
     
     @param		questions	An array containing the questions corresponding to the answers.
     
     @param		answers		An array of CFStringRefs for each of the answers.
      
     @result	A CFDictionary with values for each of the keys defined for a recovery reference:
     			
                kSecRecVersionNumber  - the version of recovery reference
                kSecRecQuestions	  - the questions
                kSecRecLocale		  - the locale of the system used to generate the answers
                kSecRecIV			  - the IV for the password wrapping (base64)
                kSecRecWrappedPassword - the wrapped password bytes (base64)
 */
    
CFDictionaryRef CF_RETURNS_RETAINED
SecWrapRecoveryPasswordWithAnswers(CFStringRef password, CFArrayRef questions, CFArrayRef answers)
__OSX_AVAILABLE_STARTING(__MAC_10_7, __IPHONE_NA); 

/*!
     @function	SecUnwrapRecoveryPasswordWithAnswers
     @abstract	Unwrap a password with a key derived from an array of answers to questions 
     
     @param		recref	    A CFDictionary containing the recovery reference as defined above.
          
     @param		answers		An array of CFStringRefs for each of the answers.
     
     @result	The unwrapped password
     
*/

CFStringRef CF_RETURNS_RETAINED
SecUnwrapRecoveryPasswordWithAnswers(CFDictionaryRef recref, CFArrayRef answers)
__OSX_AVAILABLE_STARTING(__MAC_10_7, __IPHONE_NA); 

/*!
     @function	SecCreateRecoveryPassword
     @abstract	This function creates a random password of the form:
     			T2YG-WEGQ-WVFX-A37A-I3OM-NQKV 
     
     @result	The password
     
*/
    
CFStringRef
SecCreateRecoveryPassword(void)
__OSX_AVAILABLE_STARTING(__MAC_10_7, __IPHONE_NA); 




#ifdef __cplusplus
}
#endif
#endif /* SEC_RECOVERYPASSWORD_H */


