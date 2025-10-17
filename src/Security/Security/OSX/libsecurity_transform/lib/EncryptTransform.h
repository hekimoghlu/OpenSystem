/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 4, 2023.
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
/*!
 @header EncryptTransform
 Provide the implementation class for the Encryption and Decryption 
 transforms
 
 */

#if !defined(__ENCRYPT_TRANSFORM__)
#define __ENCRYPT_TRANSFORM__ 1

#include <CommonCrypto/CommonCryptor.h>
#include <CoreFoundation/CoreFoundation.h>
#include <Security/cssmapi.h>
#include <Security/cssmapple.h>
#include <Security/cssmtype.h>
#include <Security/SecKey.h>
#include "Transform.h"
#include "TransformFactory.h"


class EncryptDecryptBase : public Transform
{
protected:
	CSSM_PADDING			m_cssm_padding;
	CSSM_ENCRYPT_MODE		m_mode;
	CSSM_KEY_PTR			m_cssm_key;			// The cssm key from the reference key
	CSSM_CC_HANDLE			m_handle;			// The context for this key either encrypt or decrypt
	Boolean					m_forEncryption;
	Boolean					m_oaep_padding;
	CFMutableDataRef		m_processedData;
	// for "single chunk" modes or paddings (i.e. OAEP) m_accumulator accumulates all the raw cleartext until EOS.
	CFMutableDataRef		m_accumulator;
    SecTransformAttributeRef inputAH;
	
	// Used to serialize CDSA setup operations for encrypt/decrypt on a given key 
	static dispatch_once_t	serializerSetUp;
	static dispatch_queue_t		serializerTransformStartingExecution;
	
	virtual void			Finalize();
	virtual Boolean 		TransformCanExecute();
	virtual CFErrorRef 		TransformStartingExecution();
	CFErrorRef				SerializedTransformStartingExecution();
	virtual void 			AttributeChanged(SecTransformAttributeRef ah, CFTypeRef value);
	
	CFDataRef				apply_oaep_padding(CFDataRef value);
	CFDataRef				remove_oaep_padding(CFDataRef value);
	
	EncryptDecryptBase(CFStringRef type);
	
	virtual 				~EncryptDecryptBase();
	
	void					SendCSSMError(CSSM_RETURN error);

public:
	// overload to return a CFDictionary that contains the state of your transform.  Values returned should be
	// serializable.  Remember that this state will be restored before SecTransformExecute is called.  Do not
	// include the transform name in your state (this will be done for you by SecTransformCopyExternalRepresentation).
	virtual CFDictionaryRef CopyState();
	
	// overload to restore the state of your transform
	virtual void 			RestoreState(CFDictionaryRef state);
	
	// your own routines
	virtual bool 			InitializeObject(SecKeyRef key, CFErrorRef *error);
	
	
};


class EncryptTransform : public EncryptDecryptBase
{
protected:
	
public:
	
	static TransformFactory* MakeTransformFactory();
	
public:
	
protected:
	EncryptTransform() ;
	
public:
	virtual 				~EncryptTransform();
	static CF_RETURNS_RETAINED SecTransformRef 	Make();
};



class DecryptTransform : public EncryptDecryptBase
{
protected:
	
public:
	
	static TransformFactory* MakeTransformFactory();
	
public:
	
protected:
	DecryptTransform();
	
public:
	virtual 				~DecryptTransform();
	static CF_RETURNS_RETAINED SecTransformRef 	Make();
};


#endif /* !__ENCRYPT_TRANSFORM__ */
